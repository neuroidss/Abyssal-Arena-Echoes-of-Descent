# server.py
import asyncio
import json
import math
import random
import time
import os
from typing import List, Dict, Set, Any, Optional, Tuple, Deque
from collections import deque, defaultdict
import numpy as np

# --- PyTorch Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence # Potential use later if needed
    PYTORCH_AVAILABLE = True
except ImportError:
    print("!!!!!!!!!!!!!!!!! PyTorch not found. Learning bots will not function. Please install PyTorch. !!!!!!!!!!!!!!!!!")
    PYTORCH_AVAILABLE = False
    # Define dummy classes/functions if PyTorch is unavailable to prevent server crashes
    class nn: Module = type('obj', (object,), {'parameters': lambda self: []})()
    class optim: Adam = None
    class torch: device = lambda x: 'cpu'; Tensor = None; long = None; tensor = None; no_grad = lambda: lambda func: func; argmax = lambda x: type('obj', (object,), {'item': lambda: 0})(); empty=lambda *a,**kw: []; cat=lambda x, **kw: []; zeros=lambda *a, **kw: []; randn=lambda *a, **kw: []
    def F_dummy(*args, **kwargs): return torch.empty(0)
    F = type('obj', (object,), {'cross_entropy': F_dummy, 'mse_loss': F_dummy, 'gelu': F_dummy})()

# --- FastAPI Imports ---
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse

# --- Configuration (Defaults & Global State) ---

DEFAULT_CONFIG = {
    # Simulation Params
    "GRID_SIZE": 35, "NUM_HC_BOTS": 2, "NUM_LEARNING_BOTS": 2, "NUM_GOALS": 10,
    "MAX_OBSTACLES_FACTOR": 0.08, "MIN_OBSTACLES_FACTOR": 0.03, "MAX_STEPS_PER_ROUND": 2000,
    "SIMULATION_SPEED_MS": 10, # Faster default for GPU
    "FREEZE_DURATION": 25,
    "VISIBILITY_RANGE": 8,
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 0.20,
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 0.70,
    "NUM_ACTIONS": 6, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal

    # Transformer Library Params (Require Reset All)
    "SEQUENCE_LENGTH": 32, # How many past steps the Transformer sees
    "MODEL_DIM": 256,      # Transformer internal dimension
    "EMBED_DIM": 32,       # Dimension for embedding individual input features
    "MODEL_DEPTH": 4,        # Number of Transformer layers
    "MODEL_HEADS": 4,        # Number of attention heads (must divide MODEL_DIM)
    "LEARNING_RATE": 1e-4,   # Learning rate for Adam optimizer
    "BATCH_SIZE": 16,        # Number of bot steps per training batch
    "LOSS_ACTION_WEIGHT": 1.0, # Weight for action prediction loss

    # Env Generation Params (Require Reset All)
    "MIN_GOAL_START_DISTANCE_FACTOR": 0.25, "MIN_BOT_START_DISTANCE_FACTOR": 0.35,
    "MIN_BOT_GOAL_DISTANCE_FACTOR": 0.20,
}
CONFIG = DEFAULT_CONFIG.copy()
CONFIG_FILE = "simulation_config.json"

# --- PyTorch Device Setup ---
if PYTORCH_AVAILABLE:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'
        gpu_status_message = f"GPU: {'Available (' + gpu_name + ')' if device.type == 'cuda' else 'Not Available (Using CPU)'}"
        if device.type == 'cuda':
            print(f"PyTorch using device: {device} ({gpu_name})")
            torch.cuda.empty_cache() # Initial clear
        else:
            print("PyTorch using device: CPU")
    except Exception as e:
        print(f"Error during PyTorch device setup: {e}")
        device = torch.device("cpu")
        gpu_status_message = "GPU: Error detecting"
else:
    device = torch.device("cpu")
    gpu_status_message = "GPU: N/A (PyTorch Missing)"


# --- Global Simulation State ---
environment: Optional['GridEnvironment'] = None
hardcoded_bots: List['HardcodedBot'] = []
learning_bots: List['LearningBot'] = []
all_bots: List[Any] = []
learning_library: Optional['LearningLibrary'] = None # PyTorch Model
optimizer: Optional[optim.Adam] = None

simulation_task: Optional[asyncio.Task] = None
is_running: bool = False
round_number: int = 0
hc_total_goals: int = 0
learning_total_goals: int = 0
_simulation_lock = asyncio.Lock() # Protects simulation state modifications

# Learning-specific state
learning_batch_buffer: List[Tuple[List[Dict], int, Dict]] = [] # Stores (sequence_features, chosen_action, next_input_features)
loss_history: Deque[float] = deque(maxlen=50) # Store recent batch losses
last_batch_loss_info: Dict = {"total": "N/A", "action": "N/A", "state": "N/A"}
last_pred_actual_example: Dict = {"predicted": "N/A", "actual": "N/A"}


# --- WebSocket Management ---
connected_clients: Set[WebSocket] = set()

# --- Utility Functions ---
def manhattan_distance(pos1: Dict[str, int], pos2: Dict[str, int]) -> int:
    if not pos1 or not pos2: return float('inf')
    return abs(pos1.get('x', 0) - pos2.get('x', 0)) + abs(pos1.get('y', 0) - pos2.get('y', 0))

def get_config_value(key: str, default_override=None):
    # Ensure config values are appropriate types after potential string loading
    val = CONFIG.get(key, default_override if default_override is not None else DEFAULT_CONFIG.get(key))
    default_val = DEFAULT_CONFIG.get(key)
    if default_val is not None:
        try: return type(default_val)(val)
        except (ValueError, TypeError): return default_val # Fallback to default type/value
    return val # Return as is if no default type info

# --- Parameter Loading/Saving ---
def load_params_from_file():
    global CONFIG
    try:
        with open(CONFIG_FILE, 'r') as f:
            loaded_params = json.load(f)
        validated_config = DEFAULT_CONFIG.copy()
        for key, default_value in DEFAULT_CONFIG.items():
            if key in loaded_params:
                 try:
                    # Attempt type conversion for robustness
                    current_type = type(default_value)
                    value = loaded_params[key]
                    if current_type == int: validated_config[key] = int(value)
                    elif current_type == float: validated_config[key] = float(value)
                    elif current_type == bool: validated_config[key] = bool(value)
                    else: validated_config[key] = value # Assume string etc.
                 except (ValueError, TypeError):
                    print(f"Warning: Type conversion failed for '{key}'. Using default.")
            # else: use default already in validated_config
        CONFIG = validated_config
        print("Parameters loaded from", CONFIG_FILE)
        return True
    except FileNotFoundError:
        print("Config file not found. Using default parameters.")
        CONFIG = DEFAULT_CONFIG.copy(); return False
    except json.JSONDecodeError:
        print("Error decoding config file. Using default parameters.")
        CONFIG = DEFAULT_CONFIG.copy(); return False
    except Exception as e:
        print(f"Error loading parameters: {e}. Using defaults.")
        CONFIG = DEFAULT_CONFIG.copy(); return False

def save_params_to_file():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        print("Parameters saved to", CONFIG_FILE)
        return True
    except Exception as e:
        print(f"Error saving parameters: {e}"); return False

def reset_params_to_default():
    global CONFIG
    CONFIG = DEFAULT_CONFIG.copy()
    print("Parameters reset to default.")

# ================================================================
# --- PyTorch Learning Library (Transformer Model) ---
# ================================================================

if PYTORCH_AVAILABLE: # Only define models if PyTorch is installed
    class InputFeatureEncoder(nn.Module):
        """Encodes the dictionary of features into a single embedding vector."""
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.visibility_range = get_config_value('VISIBILITY_RANGE', 8)
            self.num_actions = get_config_value('NUM_ACTIONS', 6)
            self.embed_dim = get_config_value('EMBED_DIM', 32)
            self.model_dim = get_config_value('MODEL_DIM', 256)
            self.opponent_types = ['Hardcoded', 'Learning', 'None'] # Keep consistent

            # Define features and their types/ranges for embedding
            self.feature_defs = {
                'wallDistance': {'type': 'scalar', 'range': get_config_value('GRID_SIZE', 35)}, # Use grid size as max practical range
                'nearestVisibleGoalDist': {'type': 'scalar', 'range': self.visibility_range + 1},
                'numVisibleGoals': {'type': 'scalar', 'range': 10}, # Max reasonable visible goals
                'nearestOpponentDist': {'type': 'scalar', 'range': self.visibility_range + 1},
                'isFrozen': {'type': 'binary'},
                'opponentIsFrozen': {'type': 'binary'},
                'opponentType': {'type': 'category', 'num_categories': len(self.opponent_types)},
                'lastAction': {'type': 'category', 'num_categories': self.num_actions + 1} # +1 for initial state (-1)
            }

            self.embedding_layers = nn.ModuleDict()
            total_feature_embed_dim = 0

            for name, props in self.feature_defs.items():
                if props['type'] == 'scalar':
                    self.embedding_layers[name] = nn.Linear(1, self.embed_dim)
                    total_feature_embed_dim += self.embed_dim
                elif props['type'] == 'binary':
                    self.embedding_layers[name] = nn.Embedding(2, self.embed_dim)
                    total_feature_embed_dim += self.embed_dim
                elif props['type'] == 'category':
                    self.embedding_layers[name] = nn.Embedding(props['num_categories'], self.embed_dim)
                    total_feature_embed_dim += self.embed_dim

            # Final projection layer
            self.projection = nn.Sequential(
                nn.Linear(total_feature_embed_dim, self.model_dim),
                nn.GELU(), # Add non-linearity
                nn.LayerNorm(self.model_dim)
            )
            print(f"InputFeatureEncoder: Total feature embed dim: {total_feature_embed_dim} -> Model dim: {self.model_dim}")

        def forward(self, feature_dict_batch: List[Dict]) -> torch.Tensor:
            batch_size = len(feature_dict_batch)
            if batch_size == 0: return torch.empty(0, self.model_dim, device=device)

            feature_tensors = defaultdict(list)

            # Prepare inputs (convert dict values to tensors on the correct device)
            for i, features in enumerate(feature_dict_batch):
                for name, props in self.feature_defs.items():
                    value = features.get(name)
                    tensor_val = None
                    if props['type'] == 'scalar':
                        norm_val = 0.0
                        if isinstance(value, (int, float)) and not math.isnan(value) and value != float('inf'):
                            norm_val = max(0.0, min(1.0, float(value) / props['range'])) if props['range'] > 0 else 0.0
                        else: norm_val = 1.0
                        tensor_val = torch.tensor([[norm_val]], dtype=torch.float32, device=device)
                    elif props['type'] == 'binary':
                        tensor_val = torch.tensor([1 if value else 0], dtype=torch.long, device=device)
                    elif props['type'] == 'category':
                        idx = -1
                        try:
                             if name == 'opponentType': idx = self.opponent_types.index(value if value else 'None')
                             elif name == 'lastAction': idx = value + 1 if isinstance(value, int) and value >= -1 else 0 # Map -1 to index 0
                        except ValueError: idx = 0 # Default to first category index if unknown
                        tensor_val = torch.tensor([idx], dtype=torch.long, device=device)

                    if tensor_val is not None:
                        feature_tensors[name].append(tensor_val)

            # Embed features in batches
            embedded_features = []
            for name, props in self.feature_defs.items():
                if name not in feature_tensors:
                    print(f"Warning: Feature '{name}' missing from batch input.")
                    # Create a default zero embedding if feature is missing
                    if props['type'] == 'scalar':
                         default_tensor = torch.zeros(batch_size, 1, device=device)
                         embedded_features.append(self.embedding_layers[name](default_tensor))
                    elif props['type'] == 'binary':
                         default_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
                         embedded_features.append(self.embedding_layers[name](default_tensor))
                    elif props['type'] == 'category':
                         default_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
                         embedded_features.append(self.embedding_layers[name](default_tensor))
                    continue


                # Stack the list of tensors for this feature into a batch tensor
                batch_tensor = torch.cat(feature_tensors[name], dim=0) # Shape: [batch_size, 1] or [batch_size]

                # Apply embedding layer
                if props['type'] == 'scalar':
                    # Linear layer expects [batch_size, input_dim]
                    embedded_features.append(self.embedding_layers[name](batch_tensor))
                else: # binary or category uses nn.Embedding expects [batch_size] or [batch_size, seq_len]
                    # We have [batch_size, 1], so squeeze last dim if needed, or just pass [batch_size]
                    if batch_tensor.ndim > 1 and batch_tensor.shape[-1] == 1:
                        batch_tensor = batch_tensor.squeeze(-1)
                    embedded_features.append(self.embedding_layers[name](batch_tensor))


            # Concatenate all feature embeddings
            concatenated_embeddings = torch.cat(embedded_features, dim=-1) # Shape: [batch_size, total_feature_embed_dim]

            # Project to the final model dimension
            projected_embedding = self.projection(concatenated_embeddings)
            return projected_embedding # Shape: [batch_size, model_dim]

    class PositionalEncoding(nn.Module):
        """ Standard sinusoidal positional encoding """
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            position = torch.arange(max_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model, device=device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            """ x: Shape [seq_len, batch_size, embedding_dim] """
            x = x + self.pe[:x.size(0)]
            return x

    class LearningLibrary(nn.Module):
        """ The main Transformer model """
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model_dim = get_config_value('MODEL_DIM', 256)
            self.nhead = get_config_value('MODEL_HEADS', 4)
            self.num_encoder_layers = get_config_value('MODEL_DEPTH', 4)
            self.num_actions = get_config_value('NUM_ACTIONS', 6)
            self.sequence_length = get_config_value('SEQUENCE_LENGTH', 32)

            if self.model_dim % self.nhead != 0:
                # Find closest divisible dimension
                valid_dim = (self.model_dim // self.nhead) * self.nhead
                if valid_dim == 0: valid_dim = self.nhead # Ensure it's at least nhead
                print(f"Warning: MODEL_DIM ({self.model_dim}) not divisible by MODEL_HEADS ({self.nhead}). Adjusting MODEL_DIM to {valid_dim}.")
                self.model_dim = valid_dim
                CONFIG['MODEL_DIM'] = valid_dim # Update global config if adjusted

            self.feature_encoder = InputFeatureEncoder(config)
            self.pos_encoder = PositionalEncoding(self.model_dim, max_len=self.sequence_length + 1)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.nhead,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

            # Prediction Heads
            self.state_prediction_head = nn.Linear(self.model_dim, self.model_dim)
            self.action_prediction_head = nn.Linear(self.model_dim, self.num_actions)

            self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
                elif isinstance(p, nn.Linear) and p.bias is not None: nn.init.zeros_(p.bias)
                elif isinstance(p, nn.Embedding): nn.init.normal_(p.weight, mean=0.0, std=0.02)

        def forward(self, input_features_batch: List[List[Dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size = len(input_features_batch)
            seq_len = len(input_features_batch[0]) if batch_size > 0 else 0
            if batch_size == 0 or seq_len != self.sequence_length:
                 print(f"Warning: Incorrect batch size ({batch_size}) or sequence length ({seq_len} vs {self.sequence_length}) in forward pass.")
                 # Return dummy tensors with correct shape but 0 batch size
                 return torch.empty(0, self.model_dim, device=device), torch.empty(0, self.num_actions, device=device)

            # Flatten batch and sequence for InputFeatureEncoder
            # Input: List[List[Dict]] -> List[Dict] of length batch_size * seq_len
            flat_features = [step_features for seq in input_features_batch for step_features in seq]

            # Encode features -> [batch_size * seq_len, model_dim]
            try:
                embedded_steps = self.feature_encoder(flat_features)
            except Exception as e:
                 print(f"Error in feature encoder: {e}")
                 import traceback; traceback.print_exc()
                 # Return dummy tensors on error
                 return torch.empty(0, self.model_dim, device=device), torch.empty(0, self.num_actions, device=device)


            # Reshape back to [batch_size, seq_len, model_dim]
            sequence_embeddings = embedded_steps.view(batch_size, seq_len, self.model_dim)

            # Add positional encoding (requires batch_first=False temporarily)
            sequence_embeddings = sequence_embeddings.transpose(0, 1) # -> [seq_len, batch, dim]
            sequence_embeddings = self.pos_encoder(sequence_embeddings)
            sequence_embeddings = sequence_embeddings.transpose(0, 1) # -> [batch, seq_len, dim]

            # Generate causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

            # Pass through Transformer Encoder
            transformer_output = self.transformer_encoder(sequence_embeddings, mask=causal_mask)

            # Use the output of the *last* token for prediction
            last_token_output = transformer_output[:, -1, :]

            # Predict
            predicted_next_state_embeddings = self.state_prediction_head(last_token_output)
            predicted_next_action_logits = self.action_prediction_head(last_token_output)

            return predicted_next_state_embeddings, predicted_next_action_logits

# ================================================================
# --- Simulation Environment (Adapted for flexible params) ---
# ================================================================
class GridEnvironment:
    # --- (Adapted from previous Python version to remove density checks) ---
    def __init__(self, config: Dict):
        self.config = config
        self.size = max(10, get_config_value('GRID_SIZE', 35))
        self.num_goals = max(1, get_config_value('NUM_GOALS', 10))
        grid_area = self.size * self.size
        self.min_obstacles = int(grid_area * get_config_value('MIN_OBSTACLES_FACTOR', 0.03))
        self.max_obstacles = int(grid_area * get_config_value('MAX_OBSTACLES_FACTOR', 0.08))
        self.num_hc_bots = max(0, get_config_value('NUM_HC_BOTS', 2))
        self.num_learning_bots = max(0, get_config_value('NUM_LEARNING_BOTS', 2))
        self.config_factors = { # Store factors separately
             'MIN_GOAL_START_DISTANCE_FACTOR': get_config_value('MIN_GOAL_START_DISTANCE_FACTOR', 0.25),
             'MIN_BOT_START_DISTANCE_FACTOR': get_config_value('MIN_BOT_START_DISTANCE_FACTOR', 0.35),
             'MIN_BOT_GOAL_DISTANCE_FACTOR': get_config_value('MIN_BOT_GOAL_DISTANCE_FACTOR', 0.20)
        }
        self.obstacles: Set[str] = set() # "x,y"
        self.goals: List[Dict] = [] # {'x': int, 'y': int, 'id': str}
        self.claimed_goals: Set[str] = set() # goal_id
        self.start_positions: List[Dict] = [] # {'x': int, 'y': int, 'type': str, 'id': str}
        self.randomize() # Initial setup

    def _pos_to_string(self, pos: Dict) -> str: return f"{pos['x']},{pos['y']}"

    def randomize(self):
        self.obstacles.clear()
        self.goals = []
        self.claimed_goals.clear()
        self.start_positions = []
        total_bots = self.num_hc_bots + self.num_learning_bots
        total_cells = self.size * self.size
        print(f"Randomizing Env: Size={self.size}x{self.size}, Goals={self.num_goals}, HC={self.num_hc_bots}, Lrn={self.num_learning_bots}.")

        max_placement_attempts = total_cells * 10 # Increase attempts
        occupied: Set[str] = set() # "x,y"

        def get_random_position() -> Dict: return {'x': random.randrange(self.size), 'y': random.randrange(self.size)}
        def is_valid_placement(pos, occupied_set, check_distances={}) -> bool:
             if not (0 <= pos['x'] < self.size and 0 <= pos['y'] < self.size): return False # Check bounds
             pos_str = self._pos_to_string(pos)
             if pos_str in occupied_set: return False
             # Simplified distance checks - allow closer placement if necessary
             if 'goalMinDist' in check_distances:
                 if any(manhattan_distance(pos, goal) < check_distances['goalMinDist'] for goal in self.goals): return False
             if 'botMinDist' in check_distances:
                 if any(manhattan_distance(pos, bot_start) < check_distances['botMinDist'] for bot_start in self.start_positions): return False
             if 'botToGoalMinDist' in check_distances:
                 if any(manhattan_distance(pos, goal) < check_distances['botToGoalMinDist'] for goal in self.goals): return False
             return True

        # 1. Place Goals
        min_goal_dist = max(1, int(self.size * self.config_factors['MIN_GOAL_START_DISTANCE_FACTOR']))
        attempts = 0; goal_id_counter = 0
        while len(self.goals) < self.num_goals and attempts < max_placement_attempts:
            attempts += 1; pos = get_random_position()
            current_min_dist = min_goal_dist if attempts < max_placement_attempts // 2 else max(1, min_goal_dist - 1)
            if is_valid_placement(pos, occupied, {'goalMinDist': current_min_dist}):
                goal = {**pos, 'id': f"G{goal_id_counter}"}; goal_id_counter += 1
                self.goals.append(goal); occupied.add(self._pos_to_string(pos))
        if len(self.goals) < self.num_goals: print(f"Warning: Placed only {len(self.goals)}/{self.num_goals} goals after {attempts} attempts.")

        # 2. Place Bots
        min_bot_dist = max(1, int(self.size * self.config_factors['MIN_BOT_START_DISTANCE_FACTOR']))
        min_bot_goal_dist = max(1, int(self.size * self.config_factors['MIN_BOT_GOAL_DISTANCE_FACTOR']))
        attempts = 0; placed_bots = 0
        while placed_bots < total_bots and attempts < max_placement_attempts:
            attempts += 1; pos = get_random_position()
            current_min_bot_dist = min_bot_dist if attempts < max_placement_attempts // 2 else max(1, min_bot_dist -1)
            current_min_bg_dist = min_bot_goal_dist if attempts < max_placement_attempts // 2 else max(1, min_bot_goal_dist -1)
            if is_valid_placement(pos, occupied, {'botMinDist': current_min_bot_dist, 'botToGoalMinDist': current_min_bg_dist}):
                 bot_type = 'Hardcoded' if placed_bots < self.num_hc_bots else 'Learning'
                 bot_id_num = placed_bots if bot_type == 'Hardcoded' else placed_bots - self.num_hc_bots
                 bot_id = f"{bot_type[0]}{bot_id_num}"
                 self.start_positions.append({**pos, 'type': bot_type, 'id': bot_id})
                 occupied.add(self._pos_to_string(pos))
                 placed_bots += 1
        if placed_bots < total_bots: print(f"Warning: Placed only {placed_bots}/{total_bots} bots after {attempts} attempts.")

        # 3. Place Obstacles
        num_obstacles_to_place = random.randint(self.min_obstacles, self.max_obstacles)
        attempts = 0
        while len(self.obstacles) < num_obstacles_to_place and attempts < max_placement_attempts:
             attempts += 1; pos = get_random_position()
             if is_valid_placement(pos, occupied): # Don't overwrite goals/starts
                  pos_str = self._pos_to_string(pos)
                  self.obstacles.add(pos_str)
        if len(self.obstacles) < num_obstacles_to_place: print(f"Warning: Placed only {len(self.obstacles)}/{num_obstacles_to_place} obstacles.")

    def is_valid(self, pos: Dict) -> bool:
        x, y = pos.get('x'), pos.get('y')
        return 0 <= x < self.size and 0 <= y < self.size and self._pos_to_string(pos) not in self.obstacles

    def get_sensory_data(self, acting_bot: Any, all_bots_list: List[Any]) -> Dict:
        # --- (Identical to previous Python version, but uses get_config_value) ---
        bot_pos = acting_bot.pos
        visibility_range = get_config_value('VISIBILITY_RANGE', 8)
        is_frozen = acting_bot.freeze_timer > 0
        min_wall_dist = min(bot_pos['x'], bot_pos['y'], self.size - 1 - bot_pos['x'], self.size - 1 - bot_pos['y'])
        nearest_visible_goal_dist = float('inf')
        num_visible_goals = 0
        visible_goals_raw = []
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals:
                  dist = manhattan_distance(bot_pos, goal)
                  if dist <= visibility_range:
                       num_visible_goals += 1
                       visible_goals_raw.append({'x': goal['x'], 'y': goal['y'], 'id': goal['id'], 'dist': dist})
                       nearest_visible_goal_dist = min(nearest_visible_goal_dist, float(dist))
        nearest_visible_goal_dist = min(nearest_visible_goal_dist, visibility_range + 1)
        nearest_opponent_dist = float('inf')
        nearest_opponent = None
        for opponent_bot in all_bots_list:
             if opponent_bot.id == acting_bot.id: continue
             dist = manhattan_distance(bot_pos, opponent_bot.pos)
             if dist <= visibility_range:
                 if dist < nearest_opponent_dist:
                     nearest_opponent_dist = float(dist)
                     nearest_opponent = opponent_bot
        nearest_opponent_dist = min(nearest_opponent_dist, visibility_range + 1)
        opponent_is_frozen = nearest_opponent.freeze_timer > 0 if nearest_opponent else False
        opponent_type = nearest_opponent.type if nearest_opponent else 'None'
        sensory_output = {
            'wallDistance': min_wall_dist,
            'nearestVisibleGoalDist': nearest_visible_goal_dist,
            'numVisibleGoals': num_visible_goals,
            'nearestOpponentDist': nearest_opponent_dist,
            'opponentIsFrozen': opponent_is_frozen,
            'opponentType': opponent_type,
            'isFrozen': is_frozen,
            '_visibleGoals': sorted(visible_goals_raw, key=lambda g: g['dist']), # Internal use
            '_nearestOpponent': nearest_opponent # Internal use
        }
        return sensory_output

    def perform_move_action(self, bot_pos: Dict, action_index: int) -> Dict:
        # --- (Identical to previous Python version) ---
        next_pos = bot_pos.copy()
        if action_index == 0: next_pos['y'] -= 1 # Up
        elif action_index == 1: next_pos['x'] -= 1 # Left
        elif action_index == 2: next_pos['x'] += 1 # Right
        elif action_index == 3: next_pos['y'] += 1 # Down
        return next_pos

    def get_adjacent_unclaimed_goal(self, bot_pos: Dict) -> Optional[Dict]:
        # --- (Identical to previous Python version) ---
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals and manhattan_distance(bot_pos, goal) == 1:
                  return goal
        return None

    def claim_goal(self, goal_id: str, bot_id: str) -> bool:
         # --- (Identical to previous Python version) ---
        if not goal_id or goal_id in self.claimed_goals: return False
        goal = next((g for g in self.goals if g['id'] == goal_id), None)
        if goal:
             self.claimed_goals.add(goal_id)
             print(f"Goal {goal_id} at ({goal['x']},{goal['y']}) claimed by {bot_id}.")
             return True
        return False

    def are_all_goals_claimed(self) -> bool:
        # --- (Identical to previous Python version) ---
        return len(self.goals) > 0 and len(self.claimed_goals) >= len(self.goals)

    def get_state_for_client(self) -> Dict:
         # --- (Identical to previous Python version) ---
        return {'size': self.size, 'goals': self.goals, 'obstacles': list(self.obstacles), 'claimedGoals': list(self.claimed_goals)}

# ================================================================
# --- Bot Implementations (Hardcoded unchanged, Learning adapted) ---
# ================================================================

class HardcodedBot:
    # --- (Identical to previous Python version) ---
    def __init__(self, start_pos: Dict, bot_id: str):
        self.id = bot_id; self.type = "Hardcoded"; self.pos = start_pos.copy()
        self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0
        self.last_move_attempt = -1; self.stuck_counter = 0
    def reset(self, start_pos: Dict):
        self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0
        self.freeze_timer = 0; self.last_move_attempt = -1; self.stuck_counter = 0
    def get_action(self, senses: Dict, env: GridEnvironment) -> int:
        if self.freeze_timer > 0: self.stuck_counter = 0; return -1
        adjacent_goal = env.get_adjacent_unclaimed_goal(self.pos)
        if adjacent_goal: self.stuck_counter = 0; self.last_move_attempt = 5; return 5
        nearest_opponent = senses.get('_nearestOpponent')
        if nearest_opponent and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen'):
            self.stuck_counter = 0; self.last_move_attempt = 4; return 4
        visible_goals = senses.get('_visibleGoals', [])
        if visible_goals:
            nearest_goal = visible_goals[0]; dx = nearest_goal['x'] - self.pos['x']; dy = nearest_goal['y'] - self.pos['y']
            preferred_moves = []
            if abs(dx) >= abs(dy):
                if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
                if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
            else:
                if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
                if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
            if abs(dx) < abs(dy):
                 if dx != 0: preferred_moves.append(2 if dx > 0 else 1)
            else:
                 if dy != 0: preferred_moves.append(3 if dy > 0 else 0)
            for move_action in preferred_moves:
                next_pos = env.perform_move_action(self.pos, move_action)
                if env.is_valid(next_pos): self.stuck_counter = 0; self.last_move_attempt = move_action; return move_action
            self.stuck_counter += 1
            valid_moves = [a for a in range(4) if env.is_valid(env.perform_move_action(self.pos, a))]
            if valid_moves:
                 reverse_action = (self.last_move_attempt + 2) % 4 if 0 <= self.last_move_attempt <= 3 else -1
                 non_reverse_valid = [m for m in valid_moves if m != reverse_action]
                 if len(non_reverse_valid) > 0 and self.stuck_counter < 5: chosen_move = random.choice(non_reverse_valid)
                 else: chosen_move = random.choice(valid_moves)
                 self.last_move_attempt = chosen_move; return chosen_move
        else: self.stuck_counter = 0
        valid_moves = [action for action in range(4) if env.is_valid(env.perform_move_action(self.pos, action))]
        if valid_moves: chosen_move = random.choice(valid_moves); self.last_move_attempt = chosen_move; return chosen_move
        else: self.last_move_attempt = -1; return -1
    def update(self, next_pos: Dict):
        self.pos = next_pos.copy(); self.steps += 1
        if self.freeze_timer > 0: self.freeze_timer -= 1
    def apply_freeze(self, duration: int):
        self.freeze_timer = max(self.freeze_timer, duration)
    def get_state_for_client(self) -> Dict:
        return {'id': self.id, 'type': self.type, 'pos': self.pos, 'freezeTimer': self.freeze_timer, 'goalsReachedThisRound': self.goals_reached_this_round, 'steps': self.steps}


class LearningBot:
    def __init__(self, start_pos: Dict, bot_id: str, config: Dict):
        self.id = bot_id
        self.type = "Learning"
        self.config = config # Store config reference
        self.pos = start_pos.copy()
        self.steps = 0
        self.goals_reached_this_round = 0
        self.freeze_timer = 0
        self.mode = "Init"
        self.last_chosen_action = -1 # Use -1 for initial state
        self.sequence_length = get_config_value('SEQUENCE_LENGTH', 32)
        self.num_actions = get_config_value('NUM_ACTIONS', 6)
        # Store history of features for Transformer input
        self.feature_history: Deque[Dict] = deque(maxlen=self.sequence_length)
        self._hc_logic_provider: Optional[HardcodedBot] = None # For rule-based exploration
        self._initial_state_features: Optional[Dict] = None # Cache initial state features for padding


    def _get_initial_state_features(self) -> Dict:
        """ Creates a default feature set for padding the history. """
        if self._initial_state_features is None:
            vis_range = get_config_value('VISIBILITY_RANGE', 8)
            grid_size = get_config_value('GRID_SIZE', 35)
            self._initial_state_features = {
                'wallDistance': min(self.pos['x'], self.pos['y'], grid_size - 1 - self.pos['x'], grid_size - 1 - self.pos['y']), # Estimate based on start
                'nearestVisibleGoalDist': vis_range + 1,
                'numVisibleGoals': 0,
                'nearestOpponentDist': vis_range + 1,
                'isFrozen': False,
                'opponentIsFrozen': False,
                'opponentType': 'None',
                'lastAction': -1
            }
        return self._initial_state_features.copy()

    def reset(self, start_pos: Dict):
        self.pos = start_pos.copy()
        self.steps = 0
        self.goals_reached_this_round = 0
        self.freeze_timer = 0
        self.mode = "Init"
        self.last_chosen_action = -1
        self.feature_history.clear()
        self._hc_logic_provider = None
        self._initial_state_features = None # Reset cached initial features


    def _get_exploration_action(self, current_senses: Dict, env: GridEnvironment) -> int:
        """ Determines the action during exploration phase. """
        use_rule_based = random.random() < get_config_value('LEARNING_BOT_RULE_EXPLORE_PERCENT', 0.7)

        if use_rule_based:
            self.mode = "Explore (Rule)"
            try:
                if not self._hc_logic_provider: self._hc_logic_provider = HardcodedBot(self.pos, "temp_hc")
                self._hc_logic_provider.pos = self.pos.copy()
                self._hc_logic_provider.freeze_timer = 0
                self._hc_logic_provider.last_move_attempt = self.last_chosen_action if 0 <= self.last_chosen_action <= 3 else -1
                self._hc_logic_provider.stuck_counter = 0
                action = self._hc_logic_provider.get_action(current_senses, env)
                if not (0 <= action < self.num_actions): action = random.randrange(4) # Fallback
            except Exception as e:
                print(f"{self.id} Rule Explore Error: {e}"); action = random.randrange(4)
        else:
            self.mode = "Explore (Random)"
            action = random.randrange(self.num_actions)

        # Basic validation of chosen exploration action feasibility
        if action == 4 and not (current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen')): action = random.randrange(4)
        elif action == 5 and not env.get_adjacent_unclaimed_goal(self.pos): action = random.randrange(4)
        elif action < 4 and not env.is_valid(env.perform_move_action(self.pos, action)):
            valid_moves = [mv for mv in range(4) if env.is_valid(env.perform_move_action(self.pos, mv))]
            action = random.choice(valid_moves) if valid_moves else -1
        return action


    def decide_action(self, current_senses: Dict, env: GridEnvironment, predicted_action_logits: Optional[torch.Tensor]) -> int:
        """ Decides the next action based on exploration or prediction. """
        if self.freeze_timer > 0:
            self.mode = "Frozen"
            # Still store features even if frozen, representing the 'frozen' state itself
            current_features = current_senses.copy()
            current_features['lastAction'] = -1 # Indicate no action was possible
            if '_visibleGoals' in current_features: del current_features['_visibleGoals']
            if '_nearestOpponent' in current_features: del current_features['_nearestOpponent']
            self.feature_history.append(current_features)
            return -1 # Cannot act

        # Store current features *before* deciding action
        current_features = current_senses.copy()
        current_features['lastAction'] = self.last_chosen_action
        if '_visibleGoals' in current_features: del current_features['_visibleGoals']
        if '_nearestOpponent' in current_features: del current_features['_nearestOpponent']
        self.feature_history.append(current_features)

        # Exploration vs Exploitation
        explore_rate = get_config_value('LEARNING_BOT_BASE_EXPLORATION_RATE', 0.2)
        if random.random() < explore_rate or predicted_action_logits is None:
            chosen_action = self._get_exploration_action(current_senses, env)
        else:
            self.mode = "Exploit"
            predicted_action = torch.argmax(predicted_action_logits).item()
            is_valid = False
            if 0 <= predicted_action <= 3: is_valid = env.is_valid(env.perform_move_action(self.pos, predicted_action))
            elif predicted_action == 4: is_valid = (current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen'))
            elif predicted_action == 5: is_valid = env.get_adjacent_unclaimed_goal(self.pos) is not None

            if is_valid:
                chosen_action = predicted_action
                self.mode = f"Exploit ({chosen_action})"
            else:
                self.mode = f"Exploit (Fallback {predicted_action})"
                chosen_action = self._get_exploration_action(current_senses, env)

        self.last_chosen_action = chosen_action # Update *after* decision
        return chosen_action

    def get_feature_sequence(self) -> List[Dict]:
        """ Returns the current feature history, padded if needed. """
        history = list(self.feature_history)
        current_len = len(history)
        if current_len < self.sequence_length:
            padding = [self._get_initial_state_features()] * (self.sequence_length - current_len)
            history = padding + history
        # Ensure it returns exactly sequence_length
        return history[-self.sequence_length:]

    def update(self, next_pos: Dict):
        """ Updates bot state after an action is performed. """
        self.pos = next_pos.copy(); self.steps += 1
        if self.freeze_timer > 0: self.freeze_timer -= 1

    def apply_freeze(self, duration: int):
        self.freeze_timer = max(self.freeze_timer, duration)

    def get_state_for_client(self) -> Dict:
        return {'id': self.id, 'type': self.type, 'pos': self.pos, 'freezeTimer': self.freeze_timer, 'goalsReachedThisRound': self.goals_reached_this_round, 'steps': self.steps, 'mode': self.mode}


# ================================================================
# --- Simulation Control Logic (Adapted for PyTorch Learning) ---
# ================================================================

async def broadcast(message: Dict):
    """Sends a JSON message to all connected clients."""
    disconnected_clients = set()
    def serialize_payload(payload):
        # --- (Identical serialization helper as before) ---
        if isinstance(payload, torch.Tensor): return payload.cpu().tolist()
        elif isinstance(payload, np.ndarray): return payload.tolist()
        elif isinstance(payload, dict): return {k: serialize_payload(v) for k, v in payload.items()}
        elif isinstance(payload, (list, tuple)): return [serialize_payload(item) for item in payload]
        elif isinstance(payload, float) and (math.isnan(payload) or math.isinf(payload)): return str(payload)
        return payload
    try:
        message_str = json.dumps(serialize_payload(message))
    except TypeError as e:
        print(f"Error serializing message: {e}"); print(f"Original message: {message}")
        message_str = json.dumps({"type": "error", "payload": "Server serialization error"})

    active_clients = list(connected_clients) # Iterate over a copy
    for client in active_clients:
        try:
             if client.client_state == WebSocketState.CONNECTED: await client.send_text(message_str)
             elif client.client_state != WebSocketState.CONNECTING: disconnected_clients.add(client)
        except WebSocketDisconnect: disconnected_clients.add(client)
        except Exception as e: print(f"Error sending to client: {e}"); disconnected_clients.add(client)
    for client in disconnected_clients: connected_clients.discard(client)

# --- Status and State Update Functions (Mostly unchanged) ---
async def send_status_update(message: str, is_error: bool = False):
    await broadcast({"type": "status_update", "payload": {"message": message, "is_error": is_error}})
async def send_gpu_status_update():
     await broadcast({"type": "gpu_status_update", "payload": {"message": gpu_status_message}})
async def send_config_update():
    await broadcast({"type": "config_update", "payload": CONFIG})
async def send_stats_update():
    avg_loss = sum(loss_history) / len(loss_history) if loss_history else float('nan')
    await broadcast({
        "type": "stats_update",
        "payload": {"roundNumber": round_number, "hcTotalGoals": hc_total_goals, "learningTotalGoals": learning_total_goals,
                    "learningAvgLoss": f"{avg_loss:.4f}" if not math.isnan(avg_loss) else "N/A"} })
async def send_library_state_update():
     avg_loss = sum(loss_history) / len(loss_history) if loss_history else float('nan')
     state_payload = {
         'lastBatchLoss': f"{last_batch_loss_info['total']} (Act: {last_batch_loss_info['action']}, State: {last_batch_loss_info['state']})",
         'avgLoss': f"{avg_loss:.4f}" if not math.isnan(avg_loss) else "N/A",
         'lossHistory': ", ".join(f"{l:.3f}" for l in loss_history) if loss_history else "N/A",
         'predictedActionL0': last_pred_actual_example.get('predicted', 'N/A'),
         'actualActionL0': last_pred_actual_example.get('actual', 'N/A'), }
     await broadcast({"type": "library_state_update", "payload": state_payload})

def check_config_needs_reset(new_config: Dict) -> Tuple[bool, bool]:
    # --- (Identical reset check logic as before) ---
    needs_full_reset = False; needs_round_reset = False
    full_reset_keys = ["GRID_SIZE", "NUM_HC_BOTS", "NUM_LEARNING_BOTS", "NUM_GOALS", "VISIBILITY_RANGE", "SEQUENCE_LENGTH", "MODEL_DIM", "EMBED_DIM", "MODEL_DEPTH", "MODEL_HEADS", "LEARNING_RATE", "BATCH_SIZE", "LOSS_ACTION_WEIGHT", "NUM_ACTIONS", "MAX_OBSTACLES_FACTOR", "MIN_OBSTACLES_FACTOR", "MIN_GOAL_START_DISTANCE_FACTOR", "MIN_BOT_START_DISTANCE_FACTOR", "MIN_BOT_GOAL_DISTANCE_FACTOR"]
    round_reset_keys = ["MAX_STEPS_PER_ROUND"]
    for key, current_value in CONFIG.items():
        new_value = new_config.get(key)
        if new_value is not None:
             try:
                current_type = type(DEFAULT_CONFIG.get(key, current_value)) # Use default config type as reference
                converted_new_value = current_type(new_value)
                if converted_new_value != current_value:
                    if key in full_reset_keys: needs_full_reset = True; break
                    elif key in round_reset_keys: needs_round_reset = True
             except (ValueError, TypeError):
                 print(f"Type mismatch for key '{key}' during reset check.")
                 if key in full_reset_keys: needs_full_reset = True; break
                 elif key in round_reset_keys: needs_round_reset = True
    if needs_full_reset: needs_round_reset = True
    return needs_full_reset, needs_round_reset

async def setup_simulation(is_full_reset_request: bool = False):
    # --- (Adapted setup logic for PyTorch) ---
    global environment, hardcoded_bots, learning_bots, all_bots, learning_library, optimizer
    global round_number, hc_total_goals, learning_total_goals, loss_history, learning_batch_buffer

    async with _simulation_lock:
        print(f"--- Setting up Simulation (Full Reset Requested: {is_full_reset_request}) ---")
        await send_status_update("Status: Initializing...")
        stop_simulation_internal(update_ui=False)

        _needs_full, _needs_round = check_config_needs_reset(CONFIG)
        perform_full_reset = is_full_reset_request or _needs_full
        perform_round_reset = perform_full_reset or _needs_round or not environment

        # Full Reset: Model, Optimizer, Scores, History
        if perform_full_reset:
            print("Performing Full Reset (Transformer Library, Optimizer, Scores, History)...")
            learning_library = None; optimizer = None
            if PYTORCH_AVAILABLE and device.type == 'cuda': torch.cuda.empty_cache()

            if PYTORCH_AVAILABLE:
                try:
                    print("Creating new Transformer Learning Library...")
                    # Re-read config values within the reset lock
                    current_config = CONFIG.copy()
                    learning_library = LearningLibrary(current_config).to(device)
                    optimizer = optim.Adam(learning_library.parameters(), lr=get_config_value('LEARNING_RATE', 1e-4))
                    print(f"Model device: {next(learning_library.parameters()).device}")
                except Exception as error:
                    print(f"CRITICAL: Failed to create Learning Library or Optimizer: {error}")
                    import traceback; traceback.print_exc()
                    await send_status_update(f"Error: Failed AI init! {error}", is_error=True)
                    learning_library = None; optimizer = None # Ensure they are None on failure
                    return False
            else:
                print("PyTorch not available. Cannot create Learning Library.")
                await send_status_update("Error: PyTorch missing!", is_error=True)
                return False # Cannot proceed without PyTorch for learning bots

            round_number = 0; hc_total_goals = 0; learning_total_goals = 0
            loss_history.clear(); learning_batch_buffer.clear()
            print("New Library/Optimizer created. Scores & history reset.")

        # Safety check (if not full reset but components missing)
        if PYTORCH_AVAILABLE and (not learning_library or not optimizer):
             print("Warning: Learning Library or Optimizer missing, forcing re-creation.")
             try:
                 current_config = CONFIG.copy()
                 learning_library = LearningLibrary(current_config).to(device)
                 optimizer = optim.Adam(learning_library.parameters(), lr=get_config_value('LEARNING_RATE', 1e-4))
                 if not perform_full_reset:
                     round_number = 0; hc_total_goals = 0; learning_total_goals = 0
                     loss_history.clear(); learning_batch_buffer.clear()
             except Exception as error:
                 print(f"CRITICAL: Forced AI component creation failed: {error}")
                 await send_status_update(f"Error: Forced AI init failed! {error}", is_error=True)
                 return False

        # Round Reset: Environment, Bots
        if perform_round_reset:
            print(f"Creating/Resetting Environment & Bots...")
            environment = None; hardcoded_bots = []; learning_bots = []; all_bots = []
            try:
                 # Pass current config to environment
                 environment = GridEnvironment(CONFIG.copy())
            except Exception as error:
                print(f"Failed to create Environment: {error}")
                await send_status_update(f"Error: Failed Env init! {error}", is_error=True)
                return False

            total_bots_required = environment.num_hc_bots + environment.num_learning_bots
            if not environment.start_positions or len(environment.start_positions) != total_bots_required:
                 print(f"Warning: Bot/StartPos count mismatch! Req: {total_bots_required}, Got: {len(environment.start_positions)}. Proceeding anyway.")

            bot_creation_error = False
            temp_learning_bots = []
            num_learning_created = 0
            for start_info in environment.start_positions:
                 bot_id = start_info['id']
                 try:
                     if start_info['type'] == 'Hardcoded':
                         hardcoded_bots.append(HardcodedBot(start_info, bot_id))
                     elif start_info['type'] == 'Learning':
                         if not PYTORCH_AVAILABLE:
                              print(f"Skipping LearningBot {bot_id} creation - PyTorch unavailable.")
                              continue # Skip learning bot if no PyTorch
                         # Pass current config to bot
                         temp_learning_bots.append(LearningBot(start_info, bot_id, CONFIG.copy()))
                         num_learning_created += 1
                 except Exception as bot_error:
                     print(f"Failed to create bot {bot_id}: {bot_error}")
                     await send_status_update(f"Error: Failed Bot init! {bot_error}", is_error=True)
                     bot_creation_error = True; break

            if bot_creation_error: return False

            # Check if number of created learning bots matches config if PyTorch IS available
            if PYTORCH_AVAILABLE and num_learning_created != environment.num_learning_bots:
                 print(f"Warning: Created {num_learning_created} learning bots, expected {environment.num_learning_bots}.")

            learning_bots = temp_learning_bots
            all_bots = [*hardcoded_bots, *learning_bots]

            if not perform_full_reset: round_number = max(1, round_number + 1)
            else: round_number = 1
            learning_batch_buffer.clear()
            print("Environment and Bots reset/created.")

        # Soft Reset (Reset bot state on existing env)
        if not perform_round_reset and environment:
            environment.claimed_goals.clear()
            for bot in all_bots:
                 start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                 if start_info: bot.reset(start_info)
                 else: print(f"Warning: No start pos for {bot.id} during soft reset."); bot.reset({'x':1,'y':1})
            learning_batch_buffer.clear()

        # Send initial state
        await broadcast({"type": "environment_update", "payload": environment.get_state_for_client() if environment else None})
        await broadcast({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_gpu_status_update()
        await send_stats_update()
        await send_library_state_update()
        await send_status_update("Status: Ready")
        print("Setup complete.")
        return True

async def reset_round_state_internal(randomize_env: bool = True):
    # --- (Reset logic mostly unchanged, ensures buffer clear) ---
    global round_number, learning_batch_buffer
    if not environment or not all_bots: return False
    async with _simulation_lock:
        if randomize_env:
            round_number += 1
            print(f"--- Starting Round {round_number} ---")
            await send_status_update(f"Status: Starting Round {round_number}...")
            try: environment.randomize()
            except Exception as error: print(f"Error randomizing env: {error}"); return False
            # Reset bots to new positions
            for bot in all_bots:
                start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                if start_info: bot.reset(start_info)
                else: print(f"Error: No start pos for {bot.id}!"); bot.reset({'x':1,'y':1})
        else: # Just reset state on current layout
            environment.claimed_goals.clear()
            for bot in all_bots:
                start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                if start_info: bot.reset(start_info)
                else: print(f"Error: No start pos for {bot.id}!"); bot.reset({'x':1,'y':1})

        learning_batch_buffer.clear() # Clear learning buffer on any round reset

        await broadcast({"type": "environment_update", "payload": environment.get_state_for_client()})
        await broadcast({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update(); await send_library_state_update()
        return True

async def perform_learning_step():
    """ Performs one optimization step using data in the buffer. """
    global learning_library, optimizer, learning_batch_buffer, loss_history, last_batch_loss_info, last_pred_actual_example

    batch_size_config = get_config_value('BATCH_SIZE', 16)
    if not PYTORCH_AVAILABLE or not learning_library or not optimizer or len(learning_batch_buffer) < batch_size_config:
        return

    # Prepare batch
    batch_data = learning_batch_buffer[:batch_size_config]
    learning_batch_buffer = learning_batch_buffer[batch_size_config:]

    # Unzip batch data
    sequence_batch = [item[0] for item in batch_data]
    actions_batch = [item[1] for item in batch_data]
    next_features_batch = [item[2] for item in batch_data]

    if not sequence_batch: return

    try:
        learning_library.train()
        optimizer.zero_grad()

        # Forward pass: Predict based on sequence before action
        predicted_next_state_embeddings, predicted_action_logits = learning_library(sequence_batch)

        # Action Prediction Loss
        target_actions = torch.tensor(actions_batch, dtype=torch.long, device=device)
        action_loss = F.cross_entropy(predicted_action_logits, target_actions)

        # Next State Prediction Loss
        with torch.no_grad():
            target_next_state_embeddings = learning_library.feature_encoder(next_features_batch)
        state_loss = F.mse_loss(predicted_next_state_embeddings, target_next_state_embeddings)

        # Combined Loss
        action_weight = get_config_value('LOSS_ACTION_WEIGHT', 1.0)
        total_loss = (action_weight * action_loss) + state_loss

        # Backward pass and Optimize
        total_loss.backward()
        # Optional: Clip gradients
        # torch.nn.utils.clip_grad_norm_(learning_library.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        loss_val = total_loss.item()
        loss_history.append(loss_val)
        last_batch_loss_info = {
            "total": f"{loss_val:.4f}",
            "action": f"{action_loss.item():.4f}",
            "state": f"{state_loss.item():.4f}"
        }
        if actions_batch:
             last_pred_actual_example["predicted"] = torch.argmax(predicted_action_logits[0]).item()
             last_pred_actual_example["actual"] = actions_batch[0]

    except Exception as e:
        print(f"!!! Error during learning step: {e} !!!")
        import traceback; traceback.print_exc()
        learning_batch_buffer.clear() # Clear buffer on error

# --- Main Simulation Loop Task (Adapted for PyTorch) ---
# server.py (Inside the simulation_loop function)

async def simulation_loop():
    global is_running, environment, all_bots, hc_total_goals, learning_total_goals, learning_batch_buffer

    print("Simulation loop started.")
    step_counter = 0
    last_learn_time = time.monotonic()

    while is_running:
        start_time = time.monotonic()
        needs_round_reset = False  # Flag to signal reset after lock release
        error_occurred = False     # Flag for critical errors

        async with _simulation_lock:
            # --- Initial checks (as before) ---
            if not is_running or not environment or not all_bots:
                 print("Simulation loop stopping: Critical components missing.")
                 is_running = False; break
            if learning_bots and PYTORCH_AVAILABLE and (not learning_library or not optimizer):
                 print("Simulation loop stopping: Learning components missing.")
                 is_running = False; break

            round_over_signal = None
            bots_state_update = []
            learning_bot_predictions = {}

            # --- Prediction Phase (as before) ---
            if learning_bots and PYTORCH_AVAILABLE:
                learning_bot_sequences = []
                learning_bot_ids_in_batch = []
                for bot in learning_bots:
                    if bot.freeze_timer <= 0:
                        seq = bot.get_feature_sequence()
                        if len(seq) == get_config_value('SEQUENCE_LENGTH'):
                            learning_bot_sequences.append(seq)
                            learning_bot_ids_in_batch.append(bot.id)

                if learning_bot_sequences:
                    try:
                        learning_library.eval()
                        with torch.no_grad():
                            _, pred_action_logits_batch = learning_library(learning_bot_sequences)
                            for i, bot_id in enumerate(learning_bot_ids_in_batch):
                                learning_bot_predictions[bot_id] = pred_action_logits_batch[i]
                    except Exception as predict_error:
                        print(f"Error during batch prediction: {predict_error}")
                        import traceback; traceback.print_exc()
                        error_occurred = True # Set error flag

            # --- Action Execution & Data Collection Phase ---
            # Check error flag before processing bots
            if not error_occurred:
                for bot in all_bots:
                    # Removed redundant error/round_over check here, handled below
                    max_steps = get_config_value('MAX_STEPS_PER_ROUND', 2000)
                    if bot.steps >= max_steps: continue

                    action = -1; next_pos = bot.pos.copy(); senses = None
                    current_sequence_for_buffer = None

                    try:
                        senses = environment.get_sensory_data(bot, all_bots)

                        if isinstance(bot, HardcodedBot):
                            action = bot.get_action(senses, environment)
                        elif isinstance(bot, LearningBot):
                            current_sequence_for_buffer = bot.get_feature_sequence()
                            predicted_logits = learning_bot_predictions.get(bot.id)
                            action = bot.decide_action(senses, environment, predicted_logits)
                        else: action = -1

                        # --- Resolve Action (as before) ---
                        if action == -1 or bot.freeze_timer > 0: next_pos = bot.pos
                        elif 0 <= action <= 3: # Move
                            intended_pos = environment.perform_move_action(bot.pos, action)
                            if environment.is_valid(intended_pos): next_pos = intended_pos
                            else: next_pos = bot.pos
                        elif action == 4: # Punch
                            next_pos = bot.pos
                            target_bot = next((ob for ob in all_bots if ob.id != bot.id and manhattan_distance(bot.pos, ob.pos) == 1 and ob.freeze_timer <= 0), None)
                            if target_bot: target_bot.apply_freeze(get_config_value('FREEZE_DURATION', 25))
                        elif action == 5: # Claim Goal
                            next_pos = bot.pos
                            adjacent_goal = environment.get_adjacent_unclaimed_goal(bot.pos)
                            if adjacent_goal and environment.claim_goal(adjacent_goal['id'], bot.id):
                                bot.goals_reached_this_round += 1
                                if bot.type == 'Hardcoded': hc_total_goals += 1
                                else: learning_total_goals += 1
                                # Send env update immediately for responsiveness
                                await broadcast({"type": "environment_update", "payload": environment.get_state_for_client()})
                                if environment.are_all_goals_claimed():
                                    # Set signal but DO NOT break the bot loop yet
                                    round_over_signal = 'goals_claimed'
                                    print(f"--- Final goal {adjacent_goal['id']} claimed by {bot.id}! Signalling round end. ---")
                        else: next_pos = bot.pos

                        # --- Store Learning Data (as before) ---
                        if isinstance(bot, LearningBot) and action != -1 and bot.freeze_timer <= 0:
                            temp_bot_state = type('obj', (object,), {'pos': next_pos, 'id': bot.id, 'freeze_timer': bot.freeze_timer})()
                            next_senses = environment.get_sensory_data(temp_bot_state, all_bots)
                            next_features = next_senses.copy()
                            next_features['lastAction'] = action
                            if '_visibleGoals' in next_features: del next_features['_visibleGoals']
                            if '_nearestOpponent' in next_features: del next_features['_nearestOpponent']
                            if current_sequence_for_buffer and len(current_sequence_for_buffer) == get_config_value('SEQUENCE_LENGTH'):
                                learning_batch_buffer.append((current_sequence_for_buffer, action, next_features))

                        # --- Update Bot State ---
                        bot.update(next_pos)
                        bots_state_update.append(bot.get_state_for_client())

                    except Exception as error:
                        print(f"Error processing turn for bot {bot.id}: {error}")
                        import traceback; traceback.print_exc()
                        error_occurred = True
                        break # Exit bot processing loop on error

                    # If round ended during this bot's turn, stop processing further bots in this step
                    if round_over_signal:
                         break

            # --- Learning Phase (as before) ---
            if not error_occurred:
                current_time = time.monotonic()
                if learning_bots and PYTORCH_AVAILABLE and (len(learning_batch_buffer) >= get_config_value('BATCH_SIZE', 16) or (current_time - last_learn_time > 1.0 and learning_batch_buffer)):
                    await perform_learning_step()
                    last_learn_time = current_time

            # --- Send Updates (as before) ---
            if not error_occurred:
                 if bots_state_update: await broadcast({"type": "bots_update", "payload": bots_state_update})
                 await send_stats_update()
                 if learning_bots and PYTORCH_AVAILABLE: await send_library_state_update()

            # --- Check Round End Conditions (Inside Lock) ---
            if not error_occurred and not round_over_signal:
                 max_steps = get_config_value('MAX_STEPS_PER_ROUND', 2000)
                 if any(b.steps >= max_steps for b in all_bots):
                      round_over_signal = 'max_steps'
                 # Check again in case last bot claimed last goal
                 elif environment.are_all_goals_claimed():
                      round_over_signal = 'goals_claimed'

            if round_over_signal:
                 needs_round_reset = True # Signal reset needed after lock release
                 reason = "All goals claimed" if round_over_signal == 'goals_claimed' else "Max steps reached"
                 print(f"--- End of Round {round_number} ({reason}) ---")
                 # Do not call reset here, just set the flag

            # If a critical error occurred, stop the simulation loop
            if error_occurred:
                is_running = False
                await send_status_update("Status: Error! Simulation stopped.", is_error=True)

        # --- End of _simulation_lock ---

        # --- Perform Round Reset (Outside Lock) ---
        if needs_round_reset and is_running: # Only reset if running and needed
             print("Attempting to start next round...")
             reset_success = await reset_round_state_internal(randomize_env=True)
             if not reset_success:
                 print("!!! Failed to reset round, stopping simulation. !!!")
                 is_running = False # Stop simulation if reset fails
                 await send_status_update("Error resetting round! Stopped.", is_error=True)
             else:
                 print("Next round setup complete.")
                 # Reset flag (although loop continues anyway)
                 needs_round_reset = False

        # --- Delay (as before) ---
        end_time = time.monotonic()
        elapsed_ms = (end_time - start_time) * 1000
        delay_s = max(0, (get_config_value('SIMULATION_SPEED_MS', 10) - elapsed_ms) / 1000.0)
        if is_running: # Only sleep if still running
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            elif step_counter % 50 == 0: # Minimal yield
                await asyncio.sleep(0.0001)

        step_counter += 1

    # --- Loop End ---
    print("Simulation loop finished.")
    # Send final status if not already errored out
    if not error_occurred:
        await send_status_update("Status: Stopped.")

def start_simulation_internal():
    # --- (Identical start logic) ---
    global simulation_task, is_running
    if is_running or (simulation_task and not simulation_task.done()): return
    print("Starting simulation task.")
    is_running = True
    simulation_task = asyncio.create_task(simulation_loop())

def stop_simulation_internal(update_ui=True):
    # --- (Identical stop logic) ---
    global is_running
    if not is_running: return
    print("Stopping simulation task...")
    is_running = False # Signal loop to stop

# ================================================================
# --- FastAPI App and WebSocket Endpoint (Adapted for PyTorch) ---
# ================================================================
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # --- (WebSocket logic largely unchanged, uses updated status/state functions) ---
    global CONFIG, is_running, simulation_task
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.client}")
    try:
        await websocket.send_json({"type": "config_update", "payload": CONFIG})
        if environment: await websocket.send_json({"type": "environment_update", "payload": environment.get_state_for_client()})
        if all_bots: await websocket.send_json({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_gpu_status_update()
        await send_stats_update(); await send_library_state_update()
        await websocket.send_json({"type": "status_update", "payload": {"message": f"Status: {'Running' if is_running else 'Ready' if environment else 'Initializing'}"}})

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            payload = message.get("payload", {})
            # print(f"Received message: {msg_type}") # Less verbose logging

            if msg_type == "start":
                if not is_running:
                    if await setup_simulation(is_full_reset_request=False): start_simulation_internal(); await send_status_update("Status: Running...")
                    else: await send_status_update("Status: Setup Failed.", is_error=True)
            elif msg_type == "stop":
                 if is_running: stop_simulation_internal(update_ui=False)
            elif msg_type == "resetRound":
                 stop_simulation_internal(update_ui=False); await setup_simulation(is_full_reset_request=False)
            elif msg_type == "resetAll":
                 stop_simulation_internal(update_ui=False); await setup_simulation(is_full_reset_request=True)
            elif msg_type == "updateParam":
                 key = payload.get("key"); value = payload.get("value")
                 if key and key in DEFAULT_CONFIG: # Check against defaults for existence and type
                      default_type = type(DEFAULT_CONFIG[key])
                      try:
                          new_value = default_type(value) # Convert to expected type
                          needs_full, needs_round = check_config_needs_reset({key: new_value})
                          # Define non-live keys (those requiring reset)
                          non_live_keys = [ k for k in DEFAULT_CONFIG if check_config_needs_reset({k:DEFAULT_CONFIG[k]})[0] or check_config_needs_reset({k:DEFAULT_CONFIG[k]})[1] ]
                          # Add keys that are live but still better to change when stopped
                          # Examples: LEARNING_BOT_BASE_EXPLORATION_RATE, LEARNING_BOT_RULE_EXPLORE_PERCENT, FREEZE_DURATION, SIMULATION_SPEED_MS
                          live_update_ok_keys = ["SIMULATION_SPEED_MS", "FREEZE_DURATION", "LEARNING_BOT_BASE_EXPLORATION_RATE", "LEARNING_BOT_RULE_EXPLORE_PERCENT"]
                          can_update_live = (key in live_update_ok_keys) and not needs_full and not needs_round

                          if is_running and not can_update_live:
                                await send_status_update(f"Status: Stop simulation to change '{key}'.")
                          else:
                                print(f"Updating config: {key} = {new_value}")
                                CONFIG[key] = new_value; await send_config_update()
                                if not is_running and (needs_full or needs_round):
                                      await send_status_update(f"Status: Config updated. {'Reset All' if needs_full else 'New Round'} required.")
                      except (ValueError, TypeError) as e: print(f"Invalid value type for param {key}: {value} - {e}")

            elif msg_type == "saveParams":
                if save_params_to_file(): await websocket.send_json({"type": "action_feedback", "payload": {"success": True, "message": "Parameters saved."}})
                else: await websocket.send_json({"type": "action_feedback", "payload": {"success": False, "message": "Error saving."}})
            elif msg_type == "loadParams":
                stop_simulation_internal(update_ui=False)
                if load_params_from_file():
                    await send_config_update()
                    await setup_simulation(is_full_reset_request=True)
                    await websocket.send_json({"type": "action_feedback", "payload": {"success": True, "message": "Parameters loaded. Reset performed."}})
                else: await websocket.send_json({"type": "action_feedback", "payload": {"success": False, "message": "Error loading/file not found."}})
            elif msg_type == "resetParams":
                stop_simulation_internal(update_ui=False)
                reset_params_to_default(); await send_config_update()
                await setup_simulation(is_full_reset_request=True)
                await websocket.send_json({"type": "action_feedback", "payload": {"success": True, "message": "Parameters reset to default. Reset performed."}})

    except WebSocketDisconnect: print(f"Client disconnected: {websocket.client}")
    except Exception as e: print(f"WebSocket Error: {e}"); import traceback; traceback.print_exc()
    finally: connected_clients.discard(websocket); print(f"Client connection closed. Remaining: {len(connected_clients)}")


@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    load_params_from_file()
    await setup_simulation(is_full_reset_request=True)

@app.on_event("shutdown")
async def shutdown_event():
    print("Server shutting down...")
    stop_simulation_internal(update_ui=False)
    if simulation_task and not simulation_task.done():
        print("Waiting for simulation task...")
        try: await asyncio.wait_for(simulation_task, timeout=2.0)
        except asyncio.TimeoutError: print("Sim task timed out, cancelling."); simulation_task.cancel()
        except Exception as e: print(f"Error waiting for sim task: {e}")
    global learning_library, optimizer; learning_library = None; optimizer = None
    if PYTORCH_AVAILABLE and device.type == 'cuda': torch.cuda.empty_cache()
    print("Shutdown complete.")

# Serve the index.html
@app.get("/", response_class=HTMLResponse)
async def get_index():
    # --- (Identical HTML serving logic) ---
    script_dir = os.path.dirname(__file__); file_path = os.path.join(script_dir, "index.html")
    try:
         with open(file_path, "r") as f: return HTMLResponse(content=f.read())
    except FileNotFoundError: return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

if __name__ == "__main__":
    print("Starting Multi-Bot Transformer Server (PyTorch)...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


