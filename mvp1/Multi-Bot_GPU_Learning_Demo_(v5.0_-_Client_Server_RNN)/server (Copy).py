# server.py
import asyncio
import json
import math
import random
import time
from typing import List, Dict, Set, Any, Optional, Tuple
import os
import numpy as np
import traceback

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
# --- End PyTorch Imports ---

# --- FastAPI Imports ---
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
# --- End FastAPI Imports ---


# --- Configuration (Defaults & Global State) ---

DEFAULT_CONFIG = {
    # Simulation Params (Require Reset Round/All handled by server)
    "GRID_SIZE": 35, "NUM_HC_BOTS": 2, "NUM_LEARNING_BOTS": 2, "NUM_GOALS": 10,
    "MAX_OBSTACLES_FACTOR": 0.08, "MIN_OBSTACLES_FACTOR": 0.03, "MAX_STEPS_PER_ROUND": 2000,
    "SIMULATION_SPEED_MS": 30, # Faster speed possible with GPU
    "FREEZE_DURATION": 25,
    "VISIBILITY_RANGE": 8,
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 0.15, # % chance explore vs exploit
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 0.60, # % of explorations using rules
    "NUM_ACTIONS": 6, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal

    # --- NEW: Cortical Learning Library Params (Require Reset All) ---
    "LIB_DEVICE": "auto", # "auto", "cuda", "cpu"
    "LIB_INPUT_EMBED_DIM": 16, # Dimension for embedding categorical features (action, type)
    "LIB_RNN_TYPE": "LSTM", # "LSTM" or "GRU"
    "LIB_HIDDEN_SIZE": 256, # Size of RNN hidden state
    "LIB_NUM_LAYERS": 2, # Number of stacked RNN layers (for hierarchy)
    "LIB_DROPOUT": 0.1, # Dropout rate within RNN layers
    "LIB_LEARNING_RATE": 0.001, # Optimizer learning rate
    "LIB_BATCH_SIZE": 1, # Set to 1 for now - step-by-step processing per bot
    "LIB_LOSS_TYPE": "MSE", # "MSE" or "L1" or "Huber" - How prediction error is calculated

    # Env Generation Params (Require Reset All)
    "MIN_GOAL_START_DISTANCE_FACTOR": 0.25, "MIN_BOT_START_DISTANCE_FACTOR": 0.35,
    "MIN_BOT_GOAL_DISTANCE_FACTOR": 0.20,
}
CONFIG = DEFAULT_CONFIG.copy()
CONFIG_FILE = "simulation_config.json"

# --- Global Simulation State ---
environment: Optional['GridEnvironment'] = None
hardcoded_bots: List['HardcodedBot'] = []
learning_bots: List['LearningBot'] = []
all_bots: List[Any] = []
cortical_library: Optional['CorticalLearningLibrary'] = None # Changed name

simulation_task: Optional[asyncio.Task] = None
is_running: bool = False
round_number: int = 0
hc_total_goals: int = 0
learning_total_goals: int = 0
_simulation_lock = asyncio.Lock() # Protects simulation state modifications

# --- WebSocket Management ---
connected_clients: Set[WebSocket] = set()

# --- Utility Functions (mostly unchanged) ---
def manhattan_distance(pos1: Dict[str, int], pos2: Dict[str, int]) -> int:
    if not pos1 or not pos2: return float('inf')
    return abs(pos1.get('x', 0) - pos2.get('x', 0)) + abs(pos1.get('y', 0) - pos2.get('y', 0))

# --- Parameter Loading/Saving (mostly unchanged) ---
def load_params_from_file():
    global CONFIG
    try:
        with open(CONFIG_FILE, 'r') as f:
            loaded_params = json.load(f)
        validated_config = DEFAULT_CONFIG.copy()
        for key, default_value in DEFAULT_CONFIG.items():
            if key in loaded_params and type(loaded_params[key]) == type(default_value):
                 validated_config[key] = loaded_params[key]
            elif key in loaded_params:
                 print(f"Warning: Loaded param '{key}' type mismatch or deprecated. Using default.")
        CONFIG = validated_config
        print("Parameters loaded from", CONFIG_FILE)
        return True
    except FileNotFoundError: CONFIG = DEFAULT_CONFIG.copy(); return False
    except json.JSONDecodeError: CONFIG = DEFAULT_CONFIG.copy(); return False
    except Exception as e: print(f"Error loading parameters: {e}. Using defaults."); CONFIG = DEFAULT_CONFIG.copy(); return False

def save_params_to_file():
    try:
        with open(CONFIG_FILE, 'w') as f: json.dump(CONFIG, f, indent=4)
        print("Parameters saved to", CONFIG_FILE); return True
    except Exception as e: print(f"Error saving parameters: {e}"); return False

def reset_params_to_default():
    global CONFIG; CONFIG = DEFAULT_CONFIG.copy(); print("Parameters reset to default.")


# ================================================================
# --- START OF AGNOSTIC GPU LIBRARY CODE (RNN Based) ---
# ================================================================

class CorticalLearningLibrary:
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._determine_device(config['LIB_DEVICE'])
        print(f"Initializing Cortical Learning Library on device: {self.device}")

        # --- Feature Encoding Parameters ---
        self.visibility_range = config['VISIBILITY_RANGE']
        self.num_actions = config['NUM_ACTIONS']
        # Define opponent types and add 'None'
        self.opponent_types = ['Hardcoded', 'Learning', 'None']
        self.opponent_type_map = {name: i for i, name in enumerate(self.opponent_types)}
        self.num_opponent_types = len(self.opponent_types)

        # Calculate input size based on encoded features
        self.scalar_feature_names = ['wallDistance', 'nearestVisibleGoalDist', 'numVisibleGoals', 'nearestOpponentDist']
        self.binary_feature_names = ['isFrozen', 'opponentIsFrozen']
        self.category_feature_names = ['lastAction', 'opponentType']

        input_size = len(self.scalar_feature_names) + len(self.binary_feature_names)
        embed_dim = config['LIB_INPUT_EMBED_DIM']
        if embed_dim > 0:
            input_size += embed_dim * len(self.category_feature_names)
        else: # Use one-hot if embed_dim is 0 (not recommended)
             input_size += self.num_actions
             input_size += self.num_opponent_types

        print(f"Calculated RNN Input Size: {input_size}")

        # --- PyTorch Model Definition ---
        self.rnn_hidden_size = config['LIB_HIDDEN_SIZE']
        self.rnn_num_layers = config['LIB_NUM_LAYERS']

        # Embeddings for categorical features
        self.action_embedding = nn.Embedding(self.num_actions + 1, embed_dim).to(self.device) if embed_dim > 0 else None # +1 for invalid action (-1)
        self.opponent_type_embedding = nn.Embedding(self.num_opponent_types, embed_dim).to(self.device) if embed_dim > 0 else None

        # RNN Layer
        rnn_class = nn.LSTM if config['LIB_RNN_TYPE'] == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True, # Expect input shape (batch, seq, feature)
            dropout=config['LIB_DROPOUT'] if self.rnn_num_layers > 1 else 0
        ).to(self.device)

        # Output Layer to predict the *next* input vector features
        self.output_layer = nn.Linear(self.rnn_hidden_size, input_size).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['LIB_LEARNING_RATE'])

        # Loss Function
        loss_type = config['LIB_LOSS_TYPE']
        if loss_type == "MSE": self.criterion = nn.MSELoss()
        elif loss_type == "L1": self.criterion = nn.L1Loss()
        elif loss_type == "Huber": self.criterion = nn.HuberLoss()
        else: print(f"Warning: Unknown loss type '{loss_type}'. Defaulting to MSE."); self.criterion = nn.MSELoss()

        # Store hidden states per bot instance {bot_id: hidden_state_tuple/tensor}
        self.hidden_states: Dict[str, Any] = {}

        # Track average loss
        self.loss_history = []
        self.max_loss_history = 100

    def _determine_device(self, requested_device: str) -> torch.device:
        if requested_device == "cuda":
            if torch.cuda.is_available(): return torch.device("cuda")
            else: print("Warning: CUDA requested but not available. Using CPU."); return torch.device("cpu")
        elif requested_device == "cpu":
            return torch.device("cpu")
        else: # auto
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parameters(self): # Helper to get all model parameters for optimizer
        params = list(self.rnn.parameters()) + list(self.output_layer.parameters())
        if self.action_embedding: params.extend(self.action_embedding.parameters())
        if self.opponent_type_embedding: params.extend(self.opponent_type_embedding.parameters())
        return params

    def encode_input(self, data_object: Dict) -> torch.Tensor:
        """Encodes the dictionary of features into a single dense tensor."""
        encoded_features = []

        # 1. Scalars (normalize)
        for name in self.scalar_feature_names:
            value = data_object.get(name, 0.0)
            if name == 'wallDistance' or name == 'nearestVisibleGoalDist' or name == 'nearestOpponentDist':
                # Treat inf/large values as max range, normalize
                norm_val = min(float(value), self.visibility_range) / max(1.0, self.visibility_range) if isinstance(value, (int, float)) and value != float('inf') else 1.0
            elif name == 'numVisibleGoals':
                norm_val = float(value) / 5.0 # Assume max ~5 visible is reasonable upper bound
            else:
                norm_val = float(value) # Fallback, might need adjustment
            encoded_features.append(norm_val)

        # 2. Binaries (0.0 or 1.0)
        for name in self.binary_feature_names:
            encoded_features.append(1.0 if data_object.get(name, False) else 0.0)

        # 3. Categories (get indices, will be embedded later if applicable)
        last_action = data_object.get('lastAction', -1)
        if not isinstance(last_action, int) or not (-1 <= last_action < self.num_actions): last_action = -1
        action_idx = torch.tensor([last_action + 1], dtype=torch.long, device=self.device) # Shift -1 to index 0

        opponent_type_name = data_object.get('opponentType', 'None')
        opponent_type_idx_val = self.opponent_type_map.get(opponent_type_name, self.opponent_type_map['None'])
        opponent_type_idx = torch.tensor([opponent_type_idx_val], dtype=torch.long, device=self.device)

        # Combine scalars and binaries into a tensor first
        scalar_binary_tensor = torch.tensor(encoded_features, dtype=torch.float32, device=self.device)

        # Embed categories or use one-hot
        if self.config['LIB_INPUT_EMBED_DIM'] > 0:
            action_embed = self.action_embedding(action_idx).squeeze(0) # Remove batch dim
            opponent_type_embed = self.opponent_type_embedding(opponent_type_idx).squeeze(0)
            combined_tensor = torch.cat((scalar_binary_tensor, action_embed, opponent_type_embed), dim=0)
        else: # One-hot fallback
            action_one_hot = nn.functional.one_hot(action_idx, num_classes=self.num_actions + 1).float().squeeze(0)
            opponent_one_hot = nn.functional.one_hot(opponent_type_idx, num_classes=self.num_opponent_types).float().squeeze(0)
            combined_tensor = torch.cat((scalar_binary_tensor, action_one_hot, opponent_one_hot), dim=0)

        # Reshape to (batch=1, seq=1, features) for RNN
        return combined_tensor.unsqueeze(0) # Add batch and seq dims

    def get_initial_hidden_state(self, bot_id: str) -> Any:
         """Returns a new zeroed hidden state."""
         batch_size = self.config['LIB_BATCH_SIZE'] # Currently 1
         h = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=self.device)
         if self.config['LIB_RNN_TYPE'] == 'LSTM':
             c = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=self.device)
             return (h, c)
         else: # GRU
             return h

    def get_persistent_hidden_state(self, bot_id: str) -> Any:
        """Retrieves the stored hidden state for a bot, or creates a new one."""
        if bot_id not in self.hidden_states:
            self.hidden_states[bot_id] = self.get_initial_hidden_state(bot_id)
        return self.hidden_states[bot_id]

    def store_hidden_state(self, bot_id: str, state: Any):
        """Stores the updated hidden state for a bot."""
        # Detach from graph to prevent history buildup across steps
        if isinstance(state, tuple): # LSTM state (h, c)
            self.hidden_states[bot_id] = (state[0].detach(), state[1].detach())
        else: # GRU state (h)
            self.hidden_states[bot_id] = state.detach()

    def reset_hidden_state(self, bot_id: str):
         """Resets the hidden state for a specific bot."""
         if bot_id in self.hidden_states:
             del self.hidden_states[bot_id] # Remove it, will be recreated on next access

    def reset_all_hidden_states(self):
        self.hidden_states.clear()

    def process_and_predict(self, input_tensor: torch.Tensor, hidden_state: Any) -> Tuple[torch.Tensor, Any]:
        """Runs the RNN forward pass to get the prediction and next hidden state."""
        # Input tensor shape: (batch=1, seq=1, features)
        # Hidden state: tuple (h, c) for LSTM, tensor h for GRU
        # Expected hidden shape by default: (num_layers, batch=1, hidden_size)
        self.rnn.train(False) # Set model to evaluation mode for prediction
        with torch.no_grad(): # No need to track gradients for prediction

            # --- START FIX for Batch=1, Seq=1 Hidden State Shape Error ---
            is_lstm = isinstance(hidden_state, tuple)
            hidden_state_for_rnn = None

            try:
                if is_lstm:
                    hx, cx = hidden_state
                    # Squeeze the batch dimension (dim 1)
                    hx_squeezed = hx.squeeze(1)
                    cx_squeezed = cx.squeeze(1)
                    # Check shapes after squeeze: should be (num_layers, hidden_size)
                    if hx_squeezed.dim() != 2 or cx_squeezed.dim() != 2:
                        print(f"Warning: Unexpected hidden state dim after squeeze. hx:{hx_squeezed.shape}, cx:{cx_squeezed.shape}")
                        # Fallback to original state if squeeze didn't work as expected
                        hidden_state_for_rnn = hidden_state
                    else:
                        hidden_state_for_rnn = (hx_squeezed, cx_squeezed)
                else: # GRU
                    hx = hidden_state
                    # Squeeze the batch dimension (dim 1)
                    hx_squeezed = hx.squeeze(1)
                    # Check shape after squeeze: should be (num_layers, hidden_size)
                    if hx_squeezed.dim() != 2:
                         print(f"Warning: Unexpected hidden state dim after squeeze. hx:{hx_squeezed.shape}")
                         hidden_state_for_rnn = hidden_state # Fallback
                    else:
                         hidden_state_for_rnn = hx_squeezed

                # Pass the potentially squeezed hidden state to the RNN
                rnn_output, next_hidden_state_processed = self.rnn(input_tensor, hidden_state_for_rnn)

                # Unsqueeze the batch dimension (dim 1) back into the output hidden state
                # The RNN output hidden state will also be 2D in this case
                if is_lstm:
                    next_hx_processed, next_cx_processed = next_hidden_state_processed
                    # Ensure they are 2D before unsqueezing
                    if next_hx_processed.dim() == 2 and next_cx_processed.dim() == 2:
                        next_hidden_state = (next_hx_processed.unsqueeze(1), next_cx_processed.unsqueeze(1))
                    else:
                        # If output wasn't 2D, maybe the squeeze wasn't needed or something else is wrong
                        print(f"Warning: Unexpected next hidden state dim. next_hx:{next_hx_processed.shape}, next_cx:{next_cx_processed.shape}")
                        next_hidden_state = next_hidden_state_processed # Return what we got
                else: # GRU
                    next_hx_processed = next_hidden_state_processed
                    if next_hx_processed.dim() == 2:
                         next_hidden_state = next_hx_processed.unsqueeze(1)
                    else:
                         print(f"Warning: Unexpected next hidden state dim. next_hx:{next_hx_processed.shape}")
                         next_hidden_state = next_hx_processed # Return what we got

            except RuntimeError as e:
                 # Catch the specific error or others during the process
                 print(f"RuntimeError during RNN forward: {e}")
                 print(f"Input shape: {input_tensor.shape}")
                 if isinstance(hidden_state, tuple): print(f"Original Hidden shapes: hx={hidden_state[0].shape}, cx={hidden_state[1].shape}")
                 else: print(f"Original Hidden shape: hx={hidden_state.shape}")
                 # Re-raise the exception after logging context
                 raise e
            except Exception as e:
                 print(f"Generic Exception during RNN forward: {e}")
                 traceback.print_exc()
                 raise e # Re-raise other errors

            # --- END FIX ---

            # rnn_output shape: (batch=1, seq=1, hidden_size)
            prediction = self.output_layer(rnn_output.squeeze(1)) # Squeeze seq dim
            # prediction shape: (batch=1, features) -> matches encoded input format

        return prediction, next_hidden_state # Return the unsqueezed state for storage

    def train_step(self, prediction_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """Performs a single training step (loss calculation, backprop, optimizer step)."""
        self.rnn.train(True) # Set model to training mode
        self.optimizer.zero_grad()

        # Ensure tensors are on the correct device and require grad for prediction
        prediction = prediction_tensor.to(self.device).requires_grad_(True)
        target = target_tensor.to(self.device)

        # Reshape if necessary (e.g., if they have batch dim but criterion doesn't expect it)
        if prediction.shape[0] == 1: prediction = prediction.squeeze(0)
        if target.shape[0] == 1: target = target.squeeze(0)

        loss = self.criterion(prediction, target)

        if torch.isnan(loss):
            print("Warning: Loss is NaN. Skipping backpropagation.")
            # Potentially log inputs/outputs that caused NaN for debugging
            return # Skip update if loss is invalid

        loss.backward()

        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Track loss
        current_loss = loss.item()
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.max_loss_history:
            self.loss_history.pop(0)

        return current_loss


    def decode_prediction(self, prediction_tensor: torch.Tensor) -> Dict:
        """Decodes the raw prediction tensor back into actionable information (primarily the action)."""
        if prediction_tensor.shape[0] == 1: prediction_tensor = prediction_tensor.squeeze(0) # Remove batch dim
        prediction_tensor = prediction_tensor.detach().cpu() # Move to CPU for numpy/logic

        decoded = {'predictedAction': None}
        current_offset = 0

        # Skip scalar and binary parts for action decoding
        num_scalar_binary = len(self.scalar_feature_names) + len(self.binary_feature_names)
        current_offset += num_scalar_binary

        # Find the most likely action
        if self.config['LIB_INPUT_EMBED_DIM'] > 0:
            action_embed_dim = self.config['LIB_INPUT_EMBED_DIM']
            predicted_action_embedding = prediction_tensor[current_offset : current_offset + action_embed_dim]
            # Find closest embedding in the learned action embedding matrix
            all_action_indices = torch.arange(self.num_actions + 1, device=self.device) # 0 to num_actions
            all_action_embeddings = self.action_embedding(all_action_indices).detach().cpu()
            # Calculate distances (e.g., cosine similarity or Euclidean)
            distances = torch.norm(all_action_embeddings - predicted_action_embedding.unsqueeze(0), dim=1)
            best_action_idx_shifted = torch.argmin(distances).item()
            predicted_action = best_action_idx_shifted - 1 # Shift back (-1 is index 0)
            current_offset += action_embed_dim
        else: # One-hot decoding
            action_one_hot_part = prediction_tensor[current_offset : current_offset + self.num_actions + 1]
            best_action_idx_shifted = torch.argmax(action_one_hot_part).item()
            predicted_action = best_action_idx_shifted - 1
            current_offset += self.num_actions + 1

        # --- Optional: Decode other features if needed ---
        # You could decode opponent type similarly, or even try to reconstruct normalized scalars,
        # but usually only the action is needed for the bot's decision.

        if -1 <= predicted_action < self.num_actions:
             decoded['predictedAction'] = predicted_action
        else: # Invalid action predicted
             # Fallback: predict a random *movement* action
             decoded['predictedAction'] = random.randrange(4)

        return decoded

    def get_average_loss(self) -> Optional[float]:
        if not self.loss_history: return None
        return sum(self.loss_history) / len(self.loss_history)

    def get_internal_state_info(self) -> Dict:
         # Provides basic info, not detailed activations like SDRs
         return {
             'device': str(self.device),
             'rnnType': self.config['LIB_RNN_TYPE'],
             'hiddenSize': self.rnn_hidden_size,
             'numLayers': self.rnn_num_layers,
             'avgLoss': f"{self.get_average_loss():.4f}" if self.get_average_loss() is not None else "N/A",
             'lossHistory': ", ".join(f"{l:.3f}" for l in self.loss_history[-10:]) # Last 10 loss points
         }

    def save_state(self, path: str = "cortical_library_state.pth"):
        """Saves the model and optimizer state."""
        state = {
            'config': self.config, # Save config used to create the model
            'model_state_dict': {
                'rnn': self.rnn.state_dict(),
                'output_layer': self.output_layer.state_dict(),
                'action_embedding': self.action_embedding.state_dict() if self.action_embedding else None,
                'opponent_type_embedding': self.opponent_type_embedding.state_dict() if self.opponent_type_embedding else None
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            # Note: Hidden states are transient and typically not saved
        }
        try:
            torch.save(state, path)
            print(f"Library state saved to {path}")
        except Exception as e:
            print(f"Error saving library state: {e}")

    def load_state(self, path: str = "cortical_library_state.pth"):
        """Loads the model and optimizer state."""
        if not os.path.exists(path):
            print(f"Error: Library state file not found at {path}")
            return False
        try:
            state = torch.load(path, map_location=self.device) # Load directly to the correct device

            # --- Compatibility Check (Basic) ---
            # Check if essential model structure params match before loading weights
            loaded_config = state.get('config', {})
            structure_keys = ['LIB_RNN_TYPE', 'LIB_HIDDEN_SIZE', 'LIB_NUM_LAYERS', 'LIB_INPUT_EMBED_DIM', 'NUM_ACTIONS'] # Add more if needed
            mismatch = False
            for key in structure_keys:
                if key not in loaded_config or loaded_config[key] != self.config.get(key):
                     print(f"Error: Config mismatch on load! Key '{key}'. Loaded: {loaded_config.get(key)}, Current: {self.config.get(key)}")
                     mismatch = True
            if mismatch:
                 print("Cannot load state due to incompatible model structure. Re-initialize or use matching config.")
                 return False
            # --- End Compatibility Check ---


            model_state = state['model_state_dict']
            self.rnn.load_state_dict(model_state['rnn'])
            self.output_layer.load_state_dict(model_state['output_layer'])
            if self.action_embedding and model_state['action_embedding']:
                 self.action_embedding.load_state_dict(model_state['action_embedding'])
            if self.opponent_type_embedding and model_state['opponent_type_embedding']:
                 self.opponent_type_embedding.load_state_dict(model_state['opponent_type_embedding'])

            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.loss_history = state.get('loss_history', [])

            print(f"Library state loaded from {path}")
            # Reset transient states after loading
            self.reset_all_hidden_states()
            return True
        except Exception as e:
            print(f"Error loading library state: {e}")
            traceback.print_exc()
            return False

    def dispose(self):
        # Clear large data structures and potentially GPU memory
        self.hidden_states.clear()
        self.loss_history = []
        del self.rnn
        del self.output_layer
        del self.action_embedding
        del self.opponent_type_embedding
        del self.optimizer
        del self.criterion
        if self.device == torch.device("cuda"):
             torch.cuda.empty_cache() # Try to release GPU memory
        print(f"Disposed Cortical Learning Library. GPU cache cleared if CUDA was used.")


# ================================================================
# --- END OF AGNOSTIC GPU LIBRARY CODE ---
# ================================================================


# ================================================================
# --- Simulation Environment (Mostly Unchanged) ---
# ================================================================
class GridEnvironment:
    # --- Constructor and methods largely unchanged ---
    # (Keep randomize, is_valid, get_sensory_data, perform_move_action,
    # get_adjacent_unclaimed_goal, claim_goal, are_all_goals_claimed,
    # get_state_for_client, _pos_to_string)

    def __init__(self, size: int, num_goals: int, min_obstacles_factor: float, max_obstacles_factor: float, num_hc_bots: int, num_learning_bots: int, config_factors: Dict):
        self.size = max(15, size)
        self.num_goals = max(1, num_goals)
        grid_area = self.size * self.size
        self.min_obstacles = int(grid_area * min_obstacles_factor)
        self.max_obstacles = int(grid_area * max_obstacles_factor)
        self.num_hc_bots = max(0, num_hc_bots)
        self.num_learning_bots = max(0, num_learning_bots)
        self.config_factors = config_factors
        self.obstacles: Set[str] = set() # "x,y"
        self.goals: List[Dict] = [] # {'x': int, 'y': int, 'id': str}
        self.claimed_goals: Set[str] = set() # goal_id
        self.start_positions: List[Dict] = [] # {'x': int, 'y': int, 'type': str, 'id': str}
        self.randomize()

    def _pos_to_string(self, pos: Dict) -> str: return f"{pos['x']},{pos['y']}"

    def randomize(self):
        self.obstacles.clear(); self.goals = []; self.claimed_goals.clear(); self.start_positions = []
        total_bots = self.num_hc_bots + self.num_learning_bots; total_cells = self.size * self.size
        required_cells = total_bots + self.num_goals; density = required_cells / total_cells if total_cells > 0 else 1.0
        print(f"Randomizing Env: Size={self.size}x{self.size}, Goals={self.num_goals}, HC={self.num_hc_bots}, Lrn={self.num_learning_bots}. Density={density:.3f}")
        if density > 0.5: print(f"Warning: High density ({(density*100):.1f}%) may make placement difficult.")
        max_placement_attempts = total_cells * 5; occupied: Set[str] = set()
        def get_random_position() -> Dict: return {'x': random.randrange(self.size), 'y': random.randrange(self.size)}
        def is_valid_placement(pos, occupied_set, check_distances={}) -> bool:
             pos_str = self._pos_to_string(pos);
             if pos_str in occupied_set: return False
             if 'goalMinDist' in check_distances:
                  if any(manhattan_distance(pos, goal) < check_distances['goalMinDist'] for goal in self.goals): return False
             if 'botMinDist' in check_distances:
                  if any(manhattan_distance(pos, bot_start) < check_distances['botMinDist'] for bot_start in self.start_positions): return False
             if 'botToGoalMinDist' in check_distances:
                  if any(manhattan_distance(pos, goal) < check_distances['botToGoalMinDist'] for goal in self.goals): return False
             return True

        min_goal_dist = max(2, int(self.size * self.config_factors['MIN_GOAL_START_DISTANCE_FACTOR'])); attempts = 0; goal_id_counter = 0
        while len(self.goals) < self.num_goals and attempts < max_placement_attempts:
            attempts += 1; pos = get_random_position()
            if is_valid_placement(pos, occupied, {'goalMinDist': min_goal_dist}):
                goal = {**pos, 'id': f"G{goal_id_counter}"}; goal_id_counter += 1
                self.goals.append(goal); occupied.add(self._pos_to_string(pos))
        if len(self.goals) < self.num_goals: print(f"Warning: Placed only {len(self.goals)}/{self.num_goals} goals.")

        min_bot_dist = max(3, int(self.size * self.config_factors['MIN_BOT_START_DISTANCE_FACTOR']))
        min_bot_goal_dist = max(3, int(self.size * self.config_factors['MIN_BOT_GOAL_DISTANCE_FACTOR']))
        attempts = 0; placed_bots = 0
        while placed_bots < total_bots and attempts < max_placement_attempts:
            attempts += 1; pos = get_random_position()
            if is_valid_placement(pos, occupied, {'botMinDist': min_bot_dist, 'botToGoalMinDist': min_bot_goal_dist}):
                 bot_type = 'Hardcoded' if placed_bots < self.num_hc_bots else 'Learning'; bot_id_num = placed_bots if bot_type == 'Hardcoded' else placed_bots - self.num_hc_bots; bot_id = f"{bot_type[0]}{bot_id_num}"
                 self.start_positions.append({**pos, 'type': bot_type, 'id': bot_id}); occupied.add(self._pos_to_string(pos)); placed_bots += 1
        if placed_bots < total_bots: print(f"CRITICAL ERROR: Placed only {placed_bots}/{total_bots} bots. Check density/constraints.")

        num_obstacles_to_place = random.randint(self.min_obstacles, self.max_obstacles); attempts = 0
        while len(self.obstacles) < num_obstacles_to_place and attempts < max_placement_attempts:
             attempts += 1; pos = get_random_position()
             if is_valid_placement(pos, occupied): pos_str = self._pos_to_string(pos); self.obstacles.add(pos_str); occupied.add(pos_str)

    def is_valid(self, pos: Dict) -> bool:
        x, y = pos.get('x'), pos.get('y'); return 0 <= x < self.size and 0 <= y < self.size and self._pos_to_string(pos) not in self.obstacles

    def get_sensory_data(self, acting_bot: Any, all_bots_list: List[Any], visibility_range: int) -> Dict:
        bot_pos = acting_bot.pos; is_frozen = acting_bot.freeze_timer > 0
        min_wall_dist = min(bot_pos['x'], bot_pos['y'], self.size - 1 - bot_pos['x'], self.size - 1 - bot_pos['y']); min_wall_dist = min(min_wall_dist, visibility_range + 1)
        nearest_visible_goal_dist = float(visibility_range + 1); num_visible_goals = 0; visible_goals_raw = []
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals:
                  dist = manhattan_distance(bot_pos, goal)
                  if dist <= visibility_range: num_visible_goals += 1; visible_goals_raw.append({'x': goal['x'], 'y': goal['y'], 'id': goal['id'], 'dist': dist}); nearest_visible_goal_dist = min(nearest_visible_goal_dist, float(dist))
        nearest_opponent_dist = float(visibility_range + 1); nearest_opponent = None
        for opponent_bot in all_bots_list:
             if opponent_bot.id == acting_bot.id: continue
             dist = manhattan_distance(bot_pos, opponent_bot.pos)
             if dist <= visibility_range:
                 if dist < nearest_opponent_dist: nearest_opponent_dist = float(dist); nearest_opponent = opponent_bot
        opponent_is_frozen = nearest_opponent.freeze_timer > 0 if nearest_opponent else False
        opponent_type = nearest_opponent.type if nearest_opponent else 'None'
        sensory_output = { 'wallDistance': min_wall_dist, 'nearestVisibleGoalDist': nearest_visible_goal_dist, 'numVisibleGoals': num_visible_goals, 'nearestOpponentDist': nearest_opponent_dist, 'opponentIsFrozen': opponent_is_frozen, 'opponentType': opponent_type, 'isFrozen': is_frozen, '_visibleGoals': sorted(visible_goals_raw, key=lambda g: g['dist']), '_nearestOpponent': nearest_opponent }
        return sensory_output

    def perform_move_action(self, bot_pos: Dict, action_index: int) -> Dict:
        next_pos = bot_pos.copy()
        if action_index == 0: next_pos['y'] -= 1 # Up
        elif action_index == 1: next_pos['x'] -= 1 # Left
        elif action_index == 2: next_pos['x'] += 1 # Right
        elif action_index == 3: next_pos['y'] += 1 # Down
        return next_pos

    def get_adjacent_unclaimed_goal(self, bot_pos: Dict) -> Optional[Dict]:
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals and manhattan_distance(bot_pos, goal) == 1: return goal
        return None

    def claim_goal(self, goal_id: str, bot_id: str) -> bool:
        if not goal_id or goal_id in self.claimed_goals: return False
        goal = next((g for g in self.goals if g['id'] == goal_id), None)
        if goal: self.claimed_goals.add(goal_id); print(f"Goal {goal_id} at ({goal['x']},{goal['y']}) claimed by {bot_id}."); return True
        return False

    def are_all_goals_claimed(self) -> bool: return len(self.goals) > 0 and len(self.claimed_goals) >= len(self.goals)
    def get_state_for_client(self) -> Dict: return { 'size': self.size, 'goals': self.goals, 'obstacles': list(self.obstacles), 'claimedGoals': list(self.claimed_goals) }


# ================================================================
# --- Bot Implementations ---
# ================================================================

class HardcodedBot:
    # --- Largely Unchanged ---
    # (Keep init, reset, get_action, update, apply_freeze, get_state_for_client)
    def __init__(self, start_pos: Dict, bot_id: str):
        self.id = bot_id; self.type = "Hardcoded"; self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0; self.last_move_attempt = -1; self.stuck_counter = 0
    def reset(self, start_pos: Dict):
        self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0; self.last_move_attempt = -1; self.stuck_counter = 0
    def get_action(self, senses: Dict, env: GridEnvironment) -> int:
        if self.freeze_timer > 0: self.stuck_counter = 0; return -1
        adjacent_goal = env.get_adjacent_unclaimed_goal(self.pos);
        if adjacent_goal: self.stuck_counter = 0; self.last_move_attempt = 5; return 5
        nearest_opponent = senses.get('_nearestOpponent');
        if nearest_opponent and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen'): self.stuck_counter = 0; self.last_move_attempt = 4; return 4
        visible_goals = senses.get('_visibleGoals', []);
        if visible_goals:
            nearest_goal = visible_goals[0]; dx = nearest_goal['x'] - self.pos['x']; dy = nearest_goal['y'] - self.pos['y']; preferred_moves = []
            if abs(dx) > abs(dy):
                if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
                if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
            else:
                if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
                if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
            for move_action in preferred_moves:
                next_pos = env.perform_move_action(self.pos, move_action)
                if env.is_valid(next_pos): self.stuck_counter = 0; self.last_move_attempt = move_action; return move_action
            self.stuck_counter += 1; sideway_moves = [0, 3] if abs(dx) > abs(dy) else [1, 2]; sideway_moves = [m for m in sideway_moves if m not in preferred_moves]; random.shuffle(sideway_moves)
            for avoid_action in sideway_moves:
                 next_pos = env.perform_move_action(self.pos, avoid_action)
                 if env.is_valid(next_pos): self.last_move_attempt = avoid_action; return avoid_action
        else: self.stuck_counter = 0
        valid_moves = [action for action in range(4) if env.is_valid(env.perform_move_action(self.pos, action))]; reverse_action = (self.last_move_attempt + 2) % 4 if 0 <= self.last_move_attempt <= 3 else -1
        if len(valid_moves) > 1 and reverse_action != -1 and self.stuck_counter < 5:
             non_reverse_moves = [m for m in valid_moves if m != reverse_action]
             if non_reverse_moves: valid_moves = non_reverse_moves
        if valid_moves: chosen_move = random.choice(valid_moves); self.last_move_attempt = chosen_move; return chosen_move
        else: self.last_move_attempt = -1; return -1
    def update(self, next_pos: Dict, chosen_action: int):
        self.pos = next_pos.copy(); self.steps += 1
        if self.freeze_timer > 0: self.freeze_timer -= 1
    def apply_freeze(self, duration: int):
        print(f"{self.id} Frozen for {duration} steps!"); self.freeze_timer = max(self.freeze_timer, duration)
    def get_state_for_client(self) -> Dict:
        return {'id': self.id, 'type': self.type, 'pos': self.pos, 'freezeTimer': self.freeze_timer, 'goalsReachedThisRound': self.goals_reached_this_round, 'steps': self.steps}


class LearningBot:
    def __init__(self, start_pos: Dict, bot_id: str, library: 'CorticalLearningLibrary', config: Dict):
        self.id = bot_id
        self.type = "Learning"
        if not library: raise ValueError("LearningBot requires valid library!")
        self.library = library
        self.config = config

        self.pos = start_pos.copy()
        self.steps = 0
        self.goals_reached_this_round = 0
        self.freeze_timer = 0

        self.last_chosen_action = -1 # Start with invalid action
        self.mode = "Init"
        self.last_loss = 0.0
        self.use_rule_based_exploration = random.random() < self.config.get('LEARNING_BOT_RULE_EXPLORE_PERCENT', 0.5)
        self._hc_logic_provider: Optional[HardcodedBot] = None # For rule-based exploration

        # --- State for Prediction/Training ---
        # Stores the prediction made *in the previous step* for the current step
        self.last_prediction_tensor: Optional[torch.Tensor] = None
        # --- End State ---


    def reset(self, start_pos: Dict):
        self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0; self.mode = "Init"; self.last_chosen_action = -1; self.last_loss = 0.0; self._hc_logic_provider = None; self.use_rule_based_exploration = random.random() < self.config.get('LEARNING_BOT_RULE_EXPLORE_PERCENT', 0.5); self.last_prediction_tensor = None
        # Reset hidden state in the library
        self.library.reset_hidden_state(self.id)

    def get_action(self, current_senses: Dict, env: GridEnvironment) -> int:
        """Chooses action via prediction or exploration AND stores prediction for next step's training."""
        chosen_action = -1

        # I. Prepare Input Data Object for *current* state
        current_input_data = current_senses.copy()
        current_input_data['lastAction'] = self.last_chosen_action # Use action that LED to current state
        if '_visibleGoals' in current_input_data: del current_input_data['_visibleGoals']
        if '_nearestOpponent' in current_input_data: del current_input_data['_nearestOpponent']

        # II. Encode Current Data to Input Tensor
        try:
            current_input_tensor = self.library.encode_input(current_input_data)
        except Exception as encode_error:
            print(f"{self.id}: Encode Error: {encode_error}"); traceback.print_exc(); self.last_chosen_action = -1; return -1

        # III. Get Prediction for *Next* Step from Library
        prediction_tensor_next = None; next_hidden_state = None
        try:
            current_hidden_state = self.library.get_persistent_hidden_state(self.id)
            prediction_tensor_next, next_hidden_state = self.library.process_and_predict(current_input_tensor, current_hidden_state)
            # Store the prediction made *for* the next step, it will be used for training *after* the next step occurs
            self.last_prediction_tensor = prediction_tensor_next.detach().clone() # Store prediction
            # Update the persistent hidden state for the *next* time this bot acts
            self.library.store_hidden_state(self.id, next_hidden_state)

        except Exception as library_error:
            print(f"{self.id}: Predictor Lib Error: {library_error}"); traceback.print_exc(); self.last_chosen_action = -1; return -1

        # IV. Decide Action (Only if NOT frozen)
        if self.freeze_timer > 0:
            self.mode = "Frozen"; chosen_action = -1
        else:
            effective_exploration_rate = self.config.get('LEARNING_BOT_BASE_EXPLORATION_RATE', 0.1)

            if random.random() < effective_exploration_rate:
                # --- Exploration Mode ---
                if self.use_rule_based_exploration:
                    self.mode = "Explore (Rule)";
                    try:
                        if not self._hc_logic_provider: self._hc_logic_provider = HardcodedBot(self.pos, "temp_hc")
                        self._hc_logic_provider.pos = self.pos.copy(); self._hc_logic_provider.freeze_timer = 0; self._hc_logic_provider.last_move_attempt = self.last_chosen_action if 0 <= self.last_chosen_action <= 3 else -1; self._hc_logic_provider.stuck_counter = 0
                        chosen_action = self._hc_logic_provider.get_action(current_senses, env)
                        if not (0 <= chosen_action < self.config['NUM_ACTIONS']): chosen_action = random.randrange(4)
                    except Exception as rule_explore_error: print(f"{self.id} Rule Explore Error: {rule_explore_error}"); chosen_action = random.randrange(4)
                else:
                    self.mode = "Explore (Random)"; chosen_action = random.randrange(self.config['NUM_ACTIONS'])
                    # Basic validation of random action
                    if chosen_action == 4 and not (current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen')): chosen_action = random.randrange(4)
                    elif chosen_action == 5 and not env.get_adjacent_unclaimed_goal(self.pos): chosen_action = random.randrange(4)
                    elif chosen_action < 4 and not env.is_valid(env.perform_move_action(self.pos, chosen_action)):
                        valid_moves = [mv for mv in range(4) if env.is_valid(env.perform_move_action(self.pos, mv))]; chosen_action = random.choice(valid_moves) if valid_moves else -1
            else:
                # --- Exploitation Mode (Use Library Prediction) ---
                try:
                    # Decode the prediction that was just made for the next step
                    decoded_prediction = self.library.decode_prediction(prediction_tensor_next)
                    predicted_action = decoded_prediction.get('predictedAction')
                    # Evaluate if the *predicted* action is valid *in the current state*
                    is_predicted_action_valid = False
                    if 0 <= predicted_action <= 3: is_predicted_action_valid = env.is_valid(env.perform_move_action(self.pos, predicted_action))
                    elif predicted_action == 4: is_predicted_action_valid = (current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen'))
                    elif predicted_action == 5: is_predicted_action_valid = env.get_adjacent_unclaimed_goal(self.pos) is not None

                    if is_predicted_action_valid:
                        self.mode = f"Exploit (Predict {predicted_action})"; chosen_action = predicted_action
                    else:
                        # Fallback logic (same as before, find *any* valid action)
                        self.mode = f"Exploit (Fallback {predicted_action}->?)"
                        adjacent_goal = env.get_adjacent_unclaimed_goal(self.pos);
                        if adjacent_goal: self.mode += " 5"; chosen_action = 5
                        elif current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen'): self.mode += " 4"; chosen_action = 4
                        else:
                            visible_goals = current_senses.get('_visibleGoals', [])
                            if visible_goals:
                                nearest_goal = visible_goals[0]; dx = nearest_goal['x'] - self.pos['x']; dy = nearest_goal['y'] - self.pos['y']; preferred_moves = []
                                if abs(dx) > abs(dy):
                                    if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
                                    if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
                                else:
                                    if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
                                    if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
                                found_move = False
                                for move in preferred_moves:
                                    if env.is_valid(env.perform_move_action(self.pos, move)): self.mode += f" {move}"; chosen_action = move; found_move = True; break
                                if not found_move:
                                    sideway_moves = [0, 3] if abs(dx) > abs(dy) else [1, 2]; sideway_moves = [m for m in sideway_moves if m not in preferred_moves]
                                    for avoid_action in sideway_moves:
                                         if env.is_valid(env.perform_move_action(self.pos, avoid_action)): self.mode += f" {avoid_action}"; chosen_action = avoid_action; found_move = True; break
                            if chosen_action == -1: # Still no action, final fallback
                                valid_moves = [a for a in range(4) if env.is_valid(env.perform_move_action(self.pos, a))]
                                if valid_moves: move = random.choice(valid_moves); self.mode += f" {move}"; chosen_action = move
                                else: self.mode += " -1"; chosen_action = -1 # Truly stuck
                except Exception as eval_error:
                    print(f"{self.id}: Predict Eval Error: {eval_error}"); traceback.print_exc(); chosen_action = random.randrange(4); self.mode = "Exploit (Error)"

        # Store the chosen action so it can be encoded in the *next* step's input
        self.last_chosen_action = chosen_action
        return chosen_action

    def update(self, next_pos: Dict, chosen_action: int):
        """Update bot's internal state (position, steps, freeze)."""
        self.pos = next_pos.copy()
        self.steps += 1
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
        # Note: Learning/training happens *after* this update, triggered by the main loop

    def train_on_last_step(self, actual_next_senses: Dict):
        """Uses the prediction made *before* the step and the actual outcome *after* the step to train."""
        if self.last_prediction_tensor is None or self.last_chosen_action == -1:
            # Cannot train if no prediction was made or no action was taken (e.g., first step, frozen)
            return 0.0 # Return 0 loss

        # I. Prepare TARGET Data Object (actual outcome of the step)
        # This uses the senses observed *after* taking 'last_chosen_action'
        target_input_data = actual_next_senses.copy()
        # The 'lastAction' for the target is the action that was just performed
        target_input_data['lastAction'] = self.last_chosen_action
        if '_visibleGoals' in target_input_data: del target_input_data['_visibleGoals']
        if '_nearestOpponent' in target_input_data: del target_input_data['_nearestOpponent']

        # II. Encode Target Data to Tensor
        try:
            target_tensor = self.library.encode_input(target_input_data)
        except Exception as encode_error:
            print(f"{self.id}: Target Encode Error: {encode_error}"); traceback.print_exc()
            self.last_prediction_tensor = None # Invalidate prediction if target failed
            return 0.0

        # III. Perform Training Step in Library
        try:
            loss = self.library.train_step(self.last_prediction_tensor, target_tensor)
            self.last_loss = loss if loss is not None else 0.0
        except Exception as train_error:
            print(f"{self.id}: Train Step Error: {train_error}"); traceback.print_exc()
            self.last_loss = 0.0 # Or maybe a large value?

        # Clear the prediction tensor used for this training step
        self.last_prediction_tensor = None

        return self.last_loss

    def apply_freeze(self, duration: int):
        print(f"{self.id} Frozen for {duration} steps!")
        self.freeze_timer = max(self.freeze_timer, duration)
        # Invalidate last prediction if frozen, as context is broken
        self.last_prediction_tensor = None
        # Reset hidden state? Optional, maybe context is lost anyway.
        # self.library.reset_hidden_state(self.id)

    def get_state_for_client(self) -> Dict:
        return {'id': self.id, 'type': self.type, 'pos': self.pos, 'freezeTimer': self.freeze_timer, 'goalsReachedThisRound': self.goals_reached_this_round, 'steps': self.steps, 'mode': self.mode, 'lastLoss': f"{self.last_loss:.4f}"} # Use loss instead of anomaly

# ================================================================
# --- Simulation Control Logic (Updated for Library) ---
# ================================================================

async def broadcast(message: Dict):
    # --- Unchanged ---
    disconnected_clients = set(); message_str = json.dumps(message)
    for client in connected_clients:
        try:
             if client.client_state == WebSocketState.CONNECTED: await client.send_text(message_str)
             elif client.client_state != WebSocketState.CONNECTING: disconnected_clients.add(client)
        except WebSocketDisconnect: disconnected_clients.add(client)
        except Exception as e: print(f"Error sending to client {client.client}: {e}"); disconnected_clients.add(client)
    for client in disconnected_clients: connected_clients.discard(client)

async def send_status_update(message: str, is_error: bool = False):
    await broadcast({"type": "status_update", "payload": {"message": message, "is_error": is_error}})

async def send_config_update():
    await broadcast({"type": "config_update", "payload": CONFIG})

async def send_stats_update():
    avg_loss = None
    if cortical_library: avg_loss = cortical_library.get_average_loss()
    await broadcast({
        "type": "stats_update", "payload": { "roundNumber": round_number, "hcTotalGoals": hc_total_goals, "learningTotalGoals": learning_total_goals, "learningAvgLoss": f"{avg_loss:.4f}" if avg_loss is not None else "N/A", }
    })

async def send_library_state_update():
     # Send basic library info (device, layers, avg loss etc)
     state_payload = cortical_library.get_internal_state_info() if cortical_library and learning_bots else None
     await broadcast({"type": "library_state_update", "payload": state_payload})


def check_config_needs_reset(new_config: Dict) -> Tuple[bool, bool]:
    # --- Updated Keys ---
    needs_full_reset = False; needs_round_reset = False
    full_reset_keys = [
        "GRID_SIZE", "NUM_HC_BOTS", "NUM_LEARNING_BOTS", "NUM_GOALS",
        "VISIBILITY_RANGE", "NUM_ACTIONS",
        "MAX_OBSTACLES_FACTOR", "MIN_OBSTACLES_FACTOR",
        "MIN_GOAL_START_DISTANCE_FACTOR", "MIN_BOT_START_DISTANCE_FACTOR", "MIN_BOT_GOAL_DISTANCE_FACTOR",
        # Library structure params require full reset
        "LIB_DEVICE", "LIB_INPUT_EMBED_DIM", "LIB_RNN_TYPE", "LIB_HIDDEN_SIZE",
        "LIB_NUM_LAYERS", "LIB_DROPOUT", "LIB_LEARNING_RATE", "LIB_BATCH_SIZE", "LIB_LOSS_TYPE"
    ]
    round_reset_keys = ["MAX_STEPS_PER_ROUND"] # Add other keys here if they only need round reset
    # Compare values
    for key, current_value in CONFIG.items():
        new_value = new_config.get(key)
        if new_value is not None and type(new_value) == type(current_value) and new_value != current_value:
            if key in full_reset_keys: needs_full_reset = True; break
            elif key in round_reset_keys: needs_round_reset = True
    if needs_full_reset: needs_round_reset = True
    return needs_full_reset, needs_round_reset


async def setup_simulation(is_full_reset_request: bool = False):
    global environment, hardcoded_bots, learning_bots, all_bots
    global cortical_library # Renamed
    global round_number, hc_total_goals, learning_total_goals

    async with _simulation_lock:
        print(f"--- Setting up Simulation (Full Reset Requested: {is_full_reset_request}) ---")
        await send_status_update("Status: Initializing...")
        stop_simulation_internal(update_ui=False)

        _needs_full, _needs_round = check_config_needs_reset(CONFIG)
        perform_full_reset = is_full_reset_request or _needs_full
        perform_round_reset = perform_full_reset or _needs_round or not environment

        # --- Perform Full Reset Actions (Library, Scores) ---
        if perform_full_reset:
            print("Performing Full Reset (Cortical Library, Scores)...")
            if cortical_library:
                try: cortical_library.dispose()
                except Exception as e: print(f"Error disposing library: {e}")
            cortical_library = None

            try:
                print("Creating new Cortical Learning Library (GPU RNN)...")
                cortical_library = CorticalLearningLibrary(CONFIG)
                # Try loading saved state if it exists and reset wasn't forced by incompatible load
                if os.path.exists("cortical_library_state.pth"):
                     print("Attempting to load saved library state...")
                     if not cortical_library.load_state():
                         print("Failed to load state or incompatible. Using fresh library.")
                     else:
                         print("Loaded saved library state successfully.")
            except Exception as error:
                print(f"CRITICAL: Failed to create Cortical Library: {error}")
                traceback.print_exc(); await send_status_update(f"Error: Failed AI init! {error}", is_error=True); return False

            round_number = 0; hc_total_goals = 0; learning_total_goals = 0
            print("New Library created/loaded. Scores reset.")

        # Safety check if library is missing
        if not cortical_library:
             print("Warning: Library missing, forcing re-creation.")
             try:
                 cortical_library = CorticalLearningLibrary(CONFIG)
                 if not perform_full_reset: round_number = 0; hc_total_goals = 0; learning_total_goals = 0
             except Exception as error:
                 print(f"CRITICAL: Forced Library creation failed: {error}"); traceback.print_exc(); await send_status_update(f"Error: Forced AI init failed! {error}", is_error=True); return False

        # --- Perform Round Reset Actions (Environment Layout, Bot Positions) ---
        if perform_round_reset:
            print(f"Creating/Resetting Environment & Bots...")
            environment = None; hardcoded_bots = []; learning_bots = []; all_bots = []
            try:
                 config_factors = { 'MIN_GOAL_START_DISTANCE_FACTOR': CONFIG['MIN_GOAL_START_DISTANCE_FACTOR'], 'MIN_BOT_START_DISTANCE_FACTOR': CONFIG['MIN_BOT_START_DISTANCE_FACTOR'], 'MIN_BOT_GOAL_DISTANCE_FACTOR': CONFIG['MIN_BOT_GOAL_DISTANCE_FACTOR'] }
                 environment = GridEnvironment(CONFIG['GRID_SIZE'], CONFIG['NUM_GOALS'], CONFIG['MIN_OBSTACLES_FACTOR'], CONFIG['MAX_OBSTACLES_FACTOR'], CONFIG['NUM_HC_BOTS'], CONFIG['NUM_LEARNING_BOTS'], config_factors)
            except Exception as error: print(f"Failed to create Environment: {error}"); await send_status_update(f"Error: Failed Env init! {error}", is_error=True); return False

            total_bots_required = CONFIG['NUM_HC_BOTS'] + CONFIG['NUM_LEARNING_BOTS']
            if not environment.start_positions or len(environment.start_positions) != total_bots_required:
                print(f"Error: Bot/StartPos count mismatch!"); await send_status_update("Error: Bot start pos mismatch!", is_error=True); return False

            bot_creation_error = False
            for start_info in environment.start_positions:
                 try:
                     if start_info['type'] == 'Hardcoded': hardcoded_bots.append(HardcodedBot(start_info, start_info['id']))
                     elif start_info['type'] == 'Learning':
                         if not cortical_library: raise ValueError("Library missing for LearningBot.")
                         learning_bots.append(LearningBot(start_info, start_info['id'], cortical_library, CONFIG))
                 except Exception as bot_error: print(f"Failed to create bot {start_info['id']}: {bot_error}"); await send_status_update(f"Error: Failed Bot init! {bot_error}", is_error=True); bot_creation_error = True; break
            if bot_creation_error: return False

            all_bots = [*hardcoded_bots, *learning_bots]
            if not perform_full_reset: round_number = max(1, round_number + 1)
            else: round_number = 1
            # Reset all hidden states in library for new round/bots
            if cortical_library: cortical_library.reset_all_hidden_states()
            print("Environment and Bots reset/created.")

        # Ensure bots are reset if round wasn't fully reset
        if not perform_round_reset and environment:
            for bot in all_bots:
                 start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                 if start_info: bot.reset(start_info)
                 else: print(f"Warning: No start pos for {bot.id} during soft reset."); bot.reset({'x':1,'y':1})
            if cortical_library: cortical_library.reset_all_hidden_states() # Also reset hidden states here

        # Send initial state
        await broadcast({"type": "environment_update", "payload": environment.get_state_for_client() if environment else None})
        await broadcast({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update()
        await send_library_state_update()
        await send_status_update("Status: Ready")
        print("Setup complete.")
        return True


async def reset_round_state_internal(randomize_env: bool = True):
    # --- Largely Unchanged, but ensures hidden states are reset ---
    global round_number
    if not environment or not all_bots: print("Error: Cannot reset round state - components missing."); await send_status_update("Error: Components missing!", is_error=True); return False

    async with _simulation_lock:
        if randomize_env:
            round_number += 1; print(f"--- Starting Round {round_number} ---"); await send_status_update(f"Status: Starting Round {round_number}...")
            try: environment.randomize()
            except Exception as error: print(f"Error randomizing environment: {error}"); await send_status_update("Error randomizing env!", is_error=True); return False
            if not environment.start_positions or len(environment.start_positions) != len(all_bots): print(f"Error: Start pos mismatch post-randomize!"); await send_status_update("Error: Bot count mismatch!", is_error=True); return False
            for bot in all_bots:
                start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                if start_info: bot.reset(start_info)
                else: print(f"Error: No start pos for bot {bot.id} after randomize!"); bot.reset({'x':1,'y':1})
            if cortical_library: cortical_library.reset_all_hidden_states() # Reset library state for new round
        else:
            environment.claimed_goals.clear()
            for bot in all_bots:
                start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                if start_info: bot.reset(start_info)
                else: print(f"Error: No start pos for bot {bot.id} during state reset!"); bot.reset({'x':1,'y':1})
            if cortical_library: cortical_library.reset_all_hidden_states() # Reset library state

        await broadcast({"type": "environment_update", "payload": environment.get_state_for_client()})
        await broadcast({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update(); await send_library_state_update()
        return True


# --- Main Simulation Loop Task (Updated for Training Step) ---
async def simulation_loop():
    global is_running, environment, all_bots, hc_total_goals, learning_total_goals
    print("Simulation loop started.")
    step_counter = 0

    while is_running:
        start_time = time.monotonic()
        step_losses = [] # Track losses within this step for averaging

        async with _simulation_lock:
            if not is_running or not environment or not all_bots or not cortical_library:
                 print("Simulation loop stopping: Components missing."); is_running = False; break

            round_over_signal = None; error_occurred = False
            bots_state_update = []

            # --- Bot Actions ---
            bot_actions: Dict[str, int] = {}
            bot_next_pos: Dict[str, Dict] = {}
            bot_senses_current: Dict[str, Dict] = {}

            for bot in all_bots:
                 if error_occurred or round_over_signal: break
                 if bot.steps >= CONFIG['MAX_STEPS_PER_ROUND']: continue

                 action = -1; senses = None
                 try:
                    senses = environment.get_sensory_data(bot, all_bots, CONFIG['VISIBILITY_RANGE'])
                    bot_senses_current[bot.id] = senses # Store senses for potential training later

                    if isinstance(bot, HardcodedBot):
                        action = bot.get_action(senses, environment)
                    elif isinstance(bot, LearningBot):
                        action = bot.get_action(senses, environment) # Get action AND store prediction
                    else: action = -1

                    bot_actions[bot.id] = action

                 except Exception as error:
                    print(f"Error getting action for bot {bot.id}: {error}"); traceback.print_exc()
                    await send_status_update(f"Error processing {bot.id}! Check server console.", is_error=True); error_occurred = True; is_running = False; break

            if error_occurred: break

            # --- Resolve Actions & Update Positions ---
            # (Two-pass to avoid order-dependent punches/moves)
            # 1. Calculate intended next positions and handle punches
            punched_bots: Set[str] = set()
            for bot in all_bots:
                action = bot_actions.get(bot.id, -1)
                current_pos = bot.pos.copy()
                next_p = current_pos # Default to staying put

                if action == -1 or bot.freeze_timer > 0: pass # No action or frozen
                elif 0 <= action <= 3: # Move intent
                    intended_pos = environment.perform_move_action(current_pos, action)
                    if environment.is_valid(intended_pos): next_p = intended_pos
                elif action == 4: # Punch intent
                    target_bot = next((ob for ob in all_bots if ob.id != bot.id and manhattan_distance(current_pos, ob.pos) == 1 and ob.freeze_timer <= 0), None)
                    if target_bot: punched_bots.add(target_bot.id)
                elif action == 5: # Claim goal intent (position doesn't change here)
                    pass

                bot_next_pos[bot.id] = next_p

            # 2. Apply punches and update bots
            goals_claimed_this_step = False
            for bot in all_bots:
                 next_pos = bot_next_pos.get(bot.id, bot.pos)
                 action = bot_actions.get(bot.id, -1)

                 if bot.id in punched_bots:
                     bot.apply_freeze(CONFIG['FREEZE_DURATION'])

                 bot.update(next_pos, action) # Update position, step count, internal freeze timer

                 if action == 5 and bot.freeze_timer <= 0: # Resolve claim goal after moving/punching
                      adjacent_goal = environment.get_adjacent_unclaimed_goal(bot.pos) # Check new position
                      if adjacent_goal and environment.claim_goal(adjacent_goal['id'], bot.id):
                            bot.goals_reached_this_round += 1
                            if bot.type == 'Hardcoded': hc_total_goals += 1
                            else: learning_total_goals += 1
                            goals_claimed_this_step = True
                            if environment.are_all_goals_claimed(): round_over_signal = 'goals_claimed'; print(f"--- Final goal {adjacent_goal['id']} claimed by {bot.id}! ---")

                 bots_state_update.append(bot.get_state_for_client())


            # --- Learning Step (After all bots moved) ---
            bot_senses_next: Dict[str, Dict] = {}
            if not error_occurred and not round_over_signal and step_counter > 0: # Allow learning after first step
                # Get senses for all bots in their *new* positions
                for bot in all_bots:
                    try:
                         bot_senses_next[bot.id] = environment.get_sensory_data(bot, all_bots, CONFIG['VISIBILITY_RANGE'])
                    except Exception as sense_error:
                         print(f"Error getting next senses for {bot.id}: {sense_error}"); error_occurred = True; break
                if error_occurred: break

                # Train each learning bot based on the prediction it made LAST step and the outcome THIS step
                for bot in learning_bots:
                     if bot.id in bot_senses_next: # Only train if we have next senses
                          try:
                              loss = bot.train_on_last_step(bot_senses_next[bot.id])
                              if loss is not None: step_losses.append(loss)
                          except Exception as train_error:
                               print(f"Error during training for {bot.id}: {train_error}"); traceback.print_exc() # Don't stop sim, just log

            if error_occurred: break

            # --- Send Updates ---
            if goals_claimed_this_step: await broadcast({"type": "environment_update", "payload": environment.get_state_for_client()})
            if bots_state_update: await broadcast({"type": "bots_update", "payload": bots_state_update})

            if step_counter % 5 == 0 or goals_claimed_this_step: # Update stats less frequently
                 await send_stats_update()
                 await send_library_state_update()

            # --- Check Round End Conditions ---
            if not round_over_signal:
                 all_bots_done = not all_bots or all(b.steps >= CONFIG['MAX_STEPS_PER_ROUND'] for b in all_bots)
                 if all_bots_done: round_over_signal = 'max_steps'
                 # Check again in case last bot claimed final goal
                 elif environment.are_all_goals_claimed(): round_over_signal = 'goals_claimed'

            if round_over_signal:
                 reason = "All goals claimed" if round_over_signal == 'goals_claimed' else "Max steps reached"
                 print(f"--- End of Round {round_number} ({reason}) ---")
                 if is_running: # Auto start next round
                     try:
                         if not await reset_round_state_internal(randomize_env=True): is_running = False; break
                         step_counter = 0 # Reset step counter for new round
                     except Exception as reset_error: print(f"Error auto-resetting round: {reset_error}"); await send_status_update("Error resetting round!", is_error=True); is_running = False; break
                 else: await send_status_update(f"Round {round_number} Finished ({reason}). Stopped."); break
            else:
                step_counter += 1

        # -- End of _simulation_lock --
        end_time = time.monotonic(); elapsed_ms = (end_time - start_time) * 1000
        delay_s = max(0.001, (CONFIG['SIMULATION_SPEED_MS'] - elapsed_ms) / 1000.0)
        await asyncio.sleep(delay_s)
    # --- Loop End ---
    print("Simulation loop finished.")
    if environment: await broadcast({"type": "environment_update", "payload": environment.get_state_for_client()})
    if all_bots: await broadcast({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
    await send_stats_update()
    if not error_occurred: await send_status_update("Status: Stopped.")

def start_simulation_internal():
    # --- Unchanged ---
    global simulation_task, is_running
    if is_running or (simulation_task and not simulation_task.done()): return
    print("Starting simulation task."); is_running = True; simulation_task = asyncio.create_task(simulation_loop())

def stop_simulation_internal(update_ui=True):
    # --- Unchanged ---
    global is_running, simulation_task
    if not is_running and (not simulation_task or simulation_task.done()): return
    print("Stopping simulation task..."); is_running = False

# ================================================================
# --- FastAPI App and WebSocket Endpoint (Minor Updates) ---
# ================================================================
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # --- Minor changes for parameter handling and loading library state ---
    global CONFIG, is_running, simulation_task
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.client}")

    try:
        # Send initial state
        await websocket.send_json({"type": "config_update", "payload": CONFIG})
        if environment: await websocket.send_json({"type": "environment_update", "payload": environment.get_state_for_client()})
        if all_bots: await websocket.send_json({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update() # Broadcast
        await send_library_state_update() # Broadcast
        await websocket.send_json({"type": "status_update", "payload": {"message": f"Status: {'Running' if is_running else 'Ready' if environment else 'Initializing'}"}})

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            payload = message.get("payload", {})
            # print(f"Received message: {msg_type}") # DEBUG

            if msg_type == "start":
                if not is_running:
                    if await setup_simulation(is_full_reset_request=False): start_simulation_internal(); await send_status_update("Status: Running...")
                    else: await send_status_update("Status: Setup Failed. Cannot start.", is_error=True)
            elif msg_type == "stop":
                 if is_running: stop_simulation_internal(update_ui=False)
            elif msg_type == "resetRound":
                 stop_simulation_internal(update_ui=False); await setup_simulation(is_full_reset_request=False)
            elif msg_type == "resetAll":
                 stop_simulation_internal(update_ui=False); await setup_simulation(is_full_reset_request=True)
            elif msg_type == "updateParam":
                 key = payload.get("key"); value = payload.get("value")
                 if key and key in CONFIG:
                      current_type = type(DEFAULT_CONFIG.get(key))
                      try:
                          if current_type == int: new_value = int(value)
                          elif current_type == float: new_value = float(value)
                          elif current_type == bool: new_value = bool(value)
                          else: new_value = value
                          if type(new_value) == current_type:
                              needs_full, needs_round = check_config_needs_reset({key: new_value})
                              # Define non-live keys more comprehensively based on full/round reset lists
                              non_live_keys = [k for k,v in DEFAULT_CONFIG.items() if check_config_needs_reset({k: 'dummy'})[0] or check_config_needs_reset({k: 'dummy'})[1]] # Crude check
                              can_update_live = not needs_full and not needs_round and key not in non_live_keys

                              if is_running and not can_update_live: await send_status_update(f"Status: Stop simulation to change '{key}'.")
                              else:
                                    print(f"Updating config: {key} = {new_value}")
                                    CONFIG[key] = new_value
                                    await send_config_update()
                                    if not is_running and (needs_full or needs_round): await send_status_update(f"Status: Config updated. {'Reset All' if needs_full else 'New Round'} required.")
                          else: print(f"Type mismatch for param {key}. Expected {current_type}, got {type(new_value)}")
                      except (ValueError, TypeError) as e: print(f"Invalid value type for param {key}: {value} - {e}")

            elif msg_type == "saveParams":
                 success_save = save_params_to_file()
                 success_lib_save = False
                 if cortical_library:
                     try: cortical_library.save_state(); success_lib_save = True
                     except Exception as e: print(f"Error saving library state: {e}")
                 msg = f"{'Params saved. ' if success_save else 'Param save failed. '}{'Library state saved.' if success_lib_save else 'Library save failed.'}"
                 await websocket.send_json({"type": "action_feedback", "payload": {"success": success_save and success_lib_save, "message": msg}})

            elif msg_type == "loadParams":
                stop_simulation_internal(update_ui=False)
                load_ok = load_params_from_file()
                if load_ok:
                    await send_config_update()
                    # Force full reset after loading params to ensure library matches loaded config
                    setup_ok = await setup_simulation(is_full_reset_request=True)
                    # Library loading happens inside setup_simulation now
                    msg = "Parameters loaded. Reset performed." + (" Library loaded." if setup_ok else " Library load/init failed.")
                    await websocket.send_json({"type": "action_feedback", "payload": {"success": load_ok and setup_ok, "message": msg}})
                else: await websocket.send_json({"type": "action_feedback", "payload": {"success": False, "message": "Error loading parameters file."}})

            elif msg_type == "resetParams":
                stop_simulation_internal(update_ui=False)
                reset_params_to_default()
                # Optionally delete saved library state file?
                if os.path.exists("cortical_library_state.pth"):
                    try: os.remove("cortical_library_state.pth"); print("Deleted saved library state file.")
                    except OSError as e: print(f"Error deleting saved library state: {e}")
                await send_config_update()
                await setup_simulation(is_full_reset_request=True) # Reset requires full setup
                await websocket.send_json({"type": "action_feedback", "payload": {"success": True, "message": "Parameters reset to default. Reset performed. Saved library state deleted (if existed)."}})

    except WebSocketDisconnect: print(f"Client disconnected: {websocket.client}")
    except Exception as e: print(f"WebSocket Error: {e}"); traceback.print_exc()
    finally: connected_clients.discard(websocket); print(f"Client connection closed: {websocket.client}. Remaining: {len(connected_clients)}")


@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    load_params_from_file()
    # Setup simulation on startup, library loading attempt happens here
    await setup_simulation(is_full_reset_request=True)

@app.on_event("shutdown")
async def shutdown_event():
    print("Server shutting down...")
    stop_simulation_internal(update_ui=False)
    if simulation_task and not simulation_task.done():
        print("Waiting for simulation task...")
        try: await asyncio.wait_for(simulation_task, timeout=2.0)
        except asyncio.TimeoutError: print("Simulation task timeout, cancelling."); simulation_task.cancel()
        except Exception as e: print(f"Error waiting for simulation task: {e}")
    if cortical_library:
        try: cortical_library.save_state() # Attempt save on shutdown
        except Exception as e: print(f"Error saving library state on shutdown: {e}")
        cortical_library.dispose()
    print("Shutdown complete.")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    script_dir = os.path.dirname(__file__); file_path = os.path.join(script_dir, "index.html")
    try:
         with open(file_path, "r") as f: return HTMLResponse(content=f.read())
    except FileNotFoundError: return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

if __name__ == "__main__":
    print("Starting Multi-Bot GPU Learning Server...")
    # GPU check is handled inside the library now
    uvicorn.run(app, host="0.0.0.0", port=8000) # Removed --reload for stability if running longer term

