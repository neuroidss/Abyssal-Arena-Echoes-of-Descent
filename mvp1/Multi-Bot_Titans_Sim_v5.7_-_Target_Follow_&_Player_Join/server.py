# server.py
# coding: utf-8
import os
import time
import math
import random
import numpy as np
from collections import deque, namedtuple
from functools import partial
import copy # For deep copying states if needed
import traceback # For detailed error logging
import sys # For recursion depth

# --- Eventlet Monkey Patching (IMPORTANT: Must be done early) ---
import eventlet
eventlet.monkey_patch() # Patch standard libraries for eventlet compatibility
print("Eventlet monkey patching applied.")

# --- Flask & SocketIO Setup ---
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Use a more specific path for templates if needed, otherwise default is fine
app = Flask(__name__, template_folder='.') # Look for index.html in the same directory
app.config['SECRET_KEY'] = os.urandom(24)
# Use eventlet for async mode
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

print("Flask and SocketIO initialized.")

# --- PyTorch Setup ---
import torch
from torch import nn, Tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter, ParameterList, ParameterDict
# Removed functional_call, vmap, grad imports as we'll use standard optimizer steps
from torch.utils._pytree import tree_map
# from tensordict import TensorDict # Removed optional dependency

# Determine device
try:
    if torch.cuda.is_available():
        # Explicitly set the device if multiple GPUs exist, otherwise default is fine
        # torch.cuda.set_device(0) # Uncomment if you need to select a specific GPU
        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        try:
            # Test tensor allocation
            _ = torch.tensor([1.0], device=device)
            torch.cuda.empty_cache() # Clear cache at start
            print("CUDA memory cache cleared and device test successful.")
        except Exception as e:
            print(f"Warning: CUDA device error during test/clear cache: {e}. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
except Exception as e:
    print(f"Error during PyTorch device setup: {e}. Falling back to CPU.")
    device = torch.device("cpu")


# ==============================================================================
# START: Integrated Titans-PyTorch NeuralMemory Code (Adapted for Server)
# ==============================================================================

# --- Helper Functions ---
def exists(v): return v is not None
def default(*args):
    for arg in args:
        if exists(arg): return arg
    return None
def l2norm(t): return F.normalize(t, dim=-1)

# --- LayerNorm (Using PyTorch Built-in with learnable gamma) ---
class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = Parameter(torch.ones(dim)) # Learnable gain, initialized to 1
        # Using PyTorch's LayerNorm without elementwise affine, applying only gamma
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # Apply LayerNorm and then the learned gamma scaling
        return self.ln(x) * self.gamma

# --- MemoryMLP (Titans-Inspired) ---
class MemoryMLP(Module):
    """ Simple MLP for the memory model """
    def __init__(self, dim, depth, expansion_factor = 2.):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        layers = []
        current_dim = dim
        for i in range(depth):
            is_last = i == (depth - 1)
            out_dim = dim if is_last else dim_hidden
            layers.append(nn.Linear(current_dim, out_dim))
            if not is_last:
                layers.append(nn.GELU()) # Activation between hidden layers
            current_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.net.modules():
             if isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight)
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# --- Neural Memory State ---
# Represents the persistent state of ONE bot's memory module
NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',        # Current sequence index processed by this memory instance
    'weights',          # Dictionary of Tensors: memory model parameters for this bot
    'optim_state',      # Dictionary: Optimizer state (e.g., momentum buffers for AdamW)
    'last_retrieved'    # Tensor: Last value retrieved *before* the update (used for anomaly calc)
])

def mem_state_detach(state: NeuralMemState):
    """ Creates a detached copy of the NeuralMemState, ensuring tensors are on their original device """
    if not isinstance(state, NeuralMemState): return state

    try:
        # Detach weights (parameters)
        detached_weights = {k: v.detach().clone().to(v.device) for k, v in state.weights.items()}

        # Detach optimizer state (more complex, needs deep copy and tensor detachment)
        detached_optim_state = {}
        if state.optim_state:
            try:
                # Use deepcopy cautiously, as it might break references needed by optimizer
                # It's safer to re-create the optimizer and load state_dict if needed
                # For detachment purpose only, copy is sufficient but ensure tensors are handled
                temp_optim_state = copy.deepcopy(state.optim_state) # Deep copy first

                def detach_and_clone_tensor(t):
                     if is_tensor(t):
                          # Ensure correct device before detaching/cloning
                          return t.detach().clone().to(t.device)
                     # Handle other types like lists or dicts within state
                     if isinstance(t, dict):
                         return {k_inner: detach_and_clone_tensor(v_inner) for k_inner, v_inner in t.items()}
                     if isinstance(t, list):
                         return [detach_and_clone_tensor(item) for item in t]
                     return t # Return non-tensor types as is

                detached_optim_state = detach_and_clone_tensor(temp_optim_state)

            except Exception as e:
                print(f"Warning: Error deep copying or detaching optimizer state: {e}. Optimizer state reset on detach.")
                detached_optim_state = {} # Reset on error

        # Detach last retrieved value
        detached_last_retrieved = state.last_retrieved.detach().clone().to(state.last_retrieved.device) if is_tensor(state.last_retrieved) else state.last_retrieved

        return NeuralMemState(
            seq_index=state.seq_index,
            weights=detached_weights,
            optim_state=detached_optim_state,
            last_retrieved=detached_last_retrieved
        )
    except Exception as e:
        print(f"CRITICAL ERROR in mem_state_detach for state at seq_index {state.seq_index}: {e}")
        traceback.print_exc()
        # Return a default/empty state or raise error
        # Returning None might be safer to indicate failure
        return None


# --- Core NeuralMemory Module (Template & Instance Holder) ---
class NeuralMemoryManager(Module): # Renamed for clarity
    """
    Holds the template architecture and provides methods to manage and update
    individual bot states using that architecture.
    """
    def __init__(
        self,
        dim,
        mem_model_depth=2,
        mem_model_expansion=2.,
        learning_rate=0.01,
        weight_decay=0.01,
        momentum_beta=0.9, # Beta1 for AdamW
        max_grad_norm=1.0,
        target_device='cpu' # The device where computations should happen
    ):
        super().__init__()
        self.dim = dim
        self.target_device = torch.device(target_device) # Ensure it's a torch.device object
        self.max_grad_norm = max_grad_norm

        # --- Template Architecture (lives on target_device) ---
        # Memory Model (e.g., MLP)
        self.memory_model_template = MemoryMLP(
            dim=dim,
            depth=mem_model_depth,
            expansion_factor=mem_model_expansion
        ).to(self.target_device)

        # Projection for calculating target value 'v' from key 'k' (input 'x')
        # This represents the "expected" output for the input stream element
        self.to_value_target_template = nn.Linear(dim, dim).to(self.target_device)
        nn.init.xavier_uniform_(self.to_value_target_template.weight)
        if self.to_value_target_template.bias is not None:
            nn.init.zeros_(self.to_value_target_template.bias)

        # --- State Management Info ---
        self.mem_param_names = list(dict(self.memory_model_template.named_parameters()).keys())

        # --- Optimizer Configuration (used to create optimizer per bot) ---
        self.optimizer_config = {
            'lr': learning_rate,
            'betas': (momentum_beta, 0.999), # AdamW betas
            'weight_decay': weight_decay
        }

        # --- Loss Function ---
        self.loss_fn = nn.MSELoss(reduction='mean')

        print(f"NeuralMemory Manager Initialized on {self.target_device}: Dim={dim}, Depth={mem_model_depth}, LR={learning_rate}, Decay={weight_decay}, Beta1={momentum_beta}")

    def get_initial_state(self) -> NeuralMemState:
        """ Returns a new, initial state dictionary for a bot, with tensors on the target_device """
        # Clone initial weights from the template model
        initial_weights = {name: p.clone().detach().to(self.target_device)
                           for name, p in self.memory_model_template.named_parameters()}
        initial_optim_state = {} # Optimizer state starts empty
        return NeuralMemState(
            seq_index=0,
            weights=initial_weights,
            optim_state=initial_optim_state,
            last_retrieved=torch.zeros(self.dim, device=self.target_device) # Initial zero retrieval
        )

    def _apply_state_to_model(self, model_instance: Module, state_weights: dict):
        """ Loads weights from the state dictionary into the provided model instance """
        # Ensure weights from state are on the same device as the model
        weights_on_device = {k: v.to(self.target_device) for k, v in state_weights.items()}
        try:
            model_instance.load_state_dict(weights_on_device, strict=True)
        except Exception as e:
            print(f"ERROR loading model state: {e}. Check architecture.")
            traceback.print_exc()
            try: # Attempt non-strict load
                model_instance.load_state_dict(weights_on_device, strict=False)
                print("Warning: Loaded model state non-strictly.")
            except Exception as e2:
                print(f"FATAL: Non-strict state loading failed: {e2}.")
                raise

    def _create_or_load_optimizer(self, model_instance: Module, state_optim_data: dict):
        """ Creates an AdamW optimizer for the model instance and loads state """
        optimizer = torch.optim.AdamW(model_instance.parameters(), **self.optimizer_config)
        if state_optim_data and 'state' in state_optim_data: # Check if state exists
            try:
                # We need to deepcopy state_dict, but ensure tensors are correctly handled for the device
                optim_state_to_load = copy.deepcopy(state_optim_data)
                def move_to_device(x):
                    if isinstance(x, torch.Tensor):
                        return x.to(self.target_device)
                    elif isinstance(x, dict):
                        return {k_inner: move_to_device(v_inner) for k_inner, v_inner in x.items()}
                    elif isinstance(x, list):
                        return [move_to_device(item) for item in x]
                    return x # Return non-tensor types as is
                optim_state_to_load = move_to_device(optim_state_to_load)

                optimizer.load_state_dict(optim_state_to_load)
                # print(f"Optimizer state loaded successfully for model instance.") # Debug
            except Exception as e:
                print(f"Warning: Failed to load optimizer state, reinitializing. Error: {e}")
                # traceback.print_exc() # More detailed debug if needed
                optimizer = torch.optim.AdamW(model_instance.parameters(), **self.optimizer_config) # Recreate on error
        # else:
            # print("No optimizer state provided or state key missing, initializing new optimizer.") # Debug
        return optimizer


    def forward_step(self, x: Tensor, current_state: NeuralMemState, detach_next_state=True):
        """
        Processes one step for a bot using its specific state.
        Args:
            x (Tensor): Input stream tensor [1, 1, dim], MUST be on self.target_device.
            current_state (NeuralMemState): The bot's current state. Tensors MUST be on self.target_device.
            detach_next_state (bool): If True, the returned next_state will have detached tensors.
        Returns:
            tuple[Tensor, NeuralMemState, float]:
                - retrieved (Tensor): Value retrieved *before* update [1, 1, dim], on target_device.
                - next_state (NeuralMemState): Updated state after learning. Tensors on target_device.
                - loss_value (float): The scalar loss calculated for this step (anomaly signal).
        """
        if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1 or x.shape[2] != self.dim:
             raise ValueError(f"Input shape error: Expected [1, 1, {self.dim}], got {x.shape}")
        if x.device != self.target_device: x = x.to(self.target_device)

        # Create temporary model instances on the target device for this step
        # Using deepcopy ensures templates are not modified
        mem_model = copy.deepcopy(self.memory_model_template).to(self.target_device)
        to_value_target = copy.deepcopy(self.to_value_target_template).to(self.target_device)

        # 1. Apply Bot's Weights to the temporary model
        self._apply_state_to_model(mem_model, current_state.weights)

        # 2. Retrieval (Inference Phase) using the temporary model
        mem_model.eval() # Set to evaluation mode for retrieval
        with torch.no_grad():
            # Process input x (key 'k') to get retrieved value M(k)
            query = x.squeeze(0) # Shape [1, dim]
            retrieved = mem_model(query) # Shape [1, dim]

        # 3. Learning Phase - update the temporary model
        mem_model.train() # Set back to training mode for learning
        optimizer = self._create_or_load_optimizer(mem_model, current_state.optim_state)
        optimizer.zero_grad() # Clear gradients before loss calculation

        # 4. Calculate Loss (Prediction Error) based on the current input 'x' (key 'k')
        key = x.squeeze(0) # Shape [1, dim]
        # Get the target value 'v' associated with the key 'k' (input 'x')
        # This target represents what the memory *should* have output for this input
        value_target = to_value_target(key) # Target 'v', Shape [1, dim]
        # Get the memory model's output *again* but this time with gradients enabled
        memory_output_for_key = mem_model(key) # Memory's prediction M(k), Shape [1, dim]

        # Calculate the loss: ||M(k) - v||^2
        loss = self.loss_fn(memory_output_for_key, value_target)
        loss_value = loss.item() # Store scalar loss value before backward pass

        # 5. Compute Gradients & Optional Clipping
        loss.backward() # Calculate gradients based on the loss
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            try:
                torch.nn.utils.clip_grad_norm_(mem_model.parameters(), self.max_grad_norm)
            except Exception as clip_err:
                print(f"Warning: Gradient clipping failed: {clip_err}")

        # 6. Update Weights using the optimizer
        optimizer.step() # Apply gradient updates to the temporary model's weights

        # 7. Capture New State from the *updated* temporary model
        # Clone weights from the *updated* mem_model
        new_weights = {name: p.clone().detach().to(self.target_device) # Ensure detached and on device
                       for name, p in mem_model.named_parameters()}

        # Get the optimizer state *after* the step
        new_optim_state_raw = optimizer.state_dict()

        # Deep copy and ensure all tensors in optimizer state are detached and on the correct device
        new_optim_state = copy.deepcopy(new_optim_state_raw)
        def process_optim_tensor(t):
            if is_tensor(t):
                return t.detach().clone().to(self.target_device)
            elif isinstance(t, dict):
                return {k_inner: process_optim_tensor(v_inner) for k_inner, v_inner in t.items()}
            elif isinstance(t, list):
                 return [process_optim_tensor(item) for item in t]
            return t # Return non-tensor types as is
        new_optim_state = process_optim_tensor(new_optim_state)


        next_seq_index = current_state.seq_index + 1
        # Store the value retrieved *before* the update (already computed and detached)
        # Ensure it's shape [dim] and on device
        last_retrieved_val = retrieved.squeeze(0).clone().detach().to(self.target_device)

        # Create the final next_state tuple
        next_state_final = NeuralMemState(
            seq_index=next_seq_index,
            weights=new_weights,
            optim_state=new_optim_state,
            last_retrieved=last_retrieved_val
        )

        # Cleanup temporary models explicitly (optional, but good practice)
        del mem_model
        del to_value_target
        del optimizer # Clean up optimizer instance for this step

        # Optionally detach the final state before returning
        if detach_next_state:
            next_state_final = mem_state_detach(next_state_final)

        # Return the value retrieved *before* update (reshaped), the new state, and the loss
        # Reshape retrieved to [1, 1, dim] to match input shape convention
        return retrieved.unsqueeze(0), next_state_final, loss_value

# ==============================================================================
# END: Integrated Titans-PyTorch NeuralMemory Code
# ==============================================================================


# ================================================================
# --- DEFAULT CONFIGURATION ---
# ================================================================
DEFAULT_CONFIG = {
    # Simulation Params
    "GRID_SIZE": 25,
    "NUM_HC_BOTS": 1,
    "NUM_LEARNING_BOTS": 3, # More learning bots to allow player takeover
    "NUM_GOALS": 5,
    "OBSTACLES_FACTOR_MIN": 0.03,
    "OBSTACLES_FACTOR_MAX": 0.08,
    "MAX_STEPS_PER_ROUND": 1000,
    "SIMULATION_SPEED_MS": 30,
    "FREEZE_DURATION": 15,
    "VISIBILITY_RANGE": 8,
    "NUM_ACTIONS": 6, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal

    # Titans-Inspired Learning Bot Params (Library)
    "LEARNING_BOT_DIM": 128,
    "LEARNING_BOT_MEM_DEPTH": 2,
    "LEARNING_BOT_LR": 0.001,
    "LEARNING_BOT_WEIGHT_DECAY": 0.01,
    "LEARNING_BOT_MOMENTUM": 0.9,
    "LEARNING_BOT_MAX_GRAD_NORM": 1.0,

    # Learning Bot Behavior (Outside Library)
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 0.15, # 15% base chance (0.0 to 1.0)
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 0.60, # 60% of exploration is rule-based (0.0 to 1.0)

    # Env Generation Params
    "MIN_GOAL_START_DISTANCE_FACTOR": 0.15,
    "MIN_BOT_START_DISTANCE_FACTOR": 0.25,
    "MIN_BOT_GOAL_DISTANCE_FACTOR": 0.15
}
current_config = copy.deepcopy(DEFAULT_CONFIG)

# --- Global State ---
bots = {} # bot_id -> bot_state_dict
players = {} # sid -> {'player_bot_id': player_bot_id, 'original_bot_id': original_learning_bot_id}
environment = None
# Create ONE NeuralMemory manager instance based on current config
neural_memory_manager = None
simulation_running = False
simulation_loop_task = None
round_number = 0
stats = {'hc_total_goals': 0, 'learning_total_goals': 0} # learning_total includes player goals
# Store direct actions from players (e.g., mobile buttons)
player_direct_actions = {} # sid -> action_code

def update_neural_memory_manager():
    """ Creates or updates the NeuralMemory manager based on current_config """
    global neural_memory_manager
    print("Updating Neural Memory Manager...")
    # If manager exists and params changed, existing bot states become incompatible
    # A full reset should be triggered by the config update handler
    try:
        neural_memory_manager = NeuralMemoryManager( # Use renamed class
            dim=current_config['LEARNING_BOT_DIM'],
            mem_model_depth=current_config['LEARNING_BOT_MEM_DEPTH'],
            learning_rate=current_config['LEARNING_BOT_LR'],
            weight_decay=current_config['LEARNING_BOT_WEIGHT_DECAY'],
            momentum_beta=current_config['LEARNING_BOT_MOMENTUM'],
            max_grad_norm=current_config['LEARNING_BOT_MAX_GRAD_NORM'],
            target_device=device # Use the globally determined device
        )
        print(f"Neural Memory Manager ready on device: {device}")
    except Exception as e:
         print(f"FATAL: Failed to create Neural Memory Manager: {e}")
         traceback.print_exc()
         neural_memory_manager = None # Indicate failure


# ================================================================
# --- Simulation Environment (Includes Obstacle Avoidance Helper) ---
# ================================================================
class GridEnvironment:
    def __init__(self, size, num_goals, obstacles_factor_range, num_hc_bots, num_learning_bots, config_factors):
        self.size = max(10, int(size))
        self.num_goals = max(0, int(num_goals))
        self.min_obstacles_factor, self.max_obstacles_factor = obstacles_factor_range
        self.num_hc_bots = max(0, int(num_hc_bots))
        self.num_learning_bots = max(0, int(num_learning_bots))
        self.config_factors = {
            'goal_dist': config_factors.get('MIN_GOAL_START_DISTANCE_FACTOR', 0.15),
            'bot_dist': config_factors.get('MIN_BOT_START_DISTANCE_FACTOR', 0.25),
            'bot_goal_dist': config_factors.get('MIN_BOT_GOAL_DISTANCE_FACTOR', 0.15)
        }
        self.obstacles = set()
        self.goals = []
        self.claimed_goals = set()
        self.start_positions = []
        self._initial_goals = []

        try:
            self.randomize()
            self._initial_goals = [{'x': g['x'], 'y': g['y'], 'id': g['id']} for g in self.goals]
        except Exception as e:
            print(f"FATAL ERROR during environment initialization: {e}")
            traceback.print_exc()
            self.size=10
            self.goals=[]
            self.obstacles=set()
            self.start_positions=[]

    def _manhattan_distance(self, pos1, pos2):
        if not pos1 or not pos2 or 'x' not in pos1 or 'y' not in pos1 or 'x' not in pos2 or 'y' not in pos2: return float('inf')
        return abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y'])

    def randomize(self):
        self.obstacles.clear()
        self.goals = []
        self.claimed_goals.clear()
        self.start_positions = []
        total_bots = self.num_hc_bots + self.num_learning_bots
        total_cells = self.size * self.size
        required_items = total_bots + self.num_goals

        print(f"Randomizing Env: Size={self.size}x{self.size}, Goals={self.num_goals}, HC={self.num_hc_bots}, Lrn={self.num_learning_bots}")

        if total_cells <= 0:
            print("ERROR: Grid size must be positive.")
            return # Changed condition
        if required_items == 0:
            print("No goals or bots to place.")
            # Still need to place obstacles if factors > 0
            # Fall through to obstacle placement

        density = required_items / total_cells if total_cells > 0 else 0
        if density > 0.7: print(f"Warning: High density ({density*100:.1f}%) - placement may fail or be slow.")

        occupied = set()
        # Increased attempts significantly for robustness with high density/constraints
        max_placement_attempts = max(required_items * 50, total_cells * 10)

        def is_valid_placement(pos_tuple, occupied_set, check_dists={}):
            # Check bounds first
            if not (0 <= pos_tuple[0] < self.size and 0 <= pos_tuple[1] < self.size): return False
            # Check direct occupation
            if pos_tuple in occupied_set: return False
            # Check distance constraints
            pos_dict = {'x': pos_tuple[0], 'y': pos_tuple[1]}
            goal_min = check_dists.get('goal_min_dist', 0)
            bot_min = check_dists.get('bot_min_dist', 0)
            bot_goal_min = check_dists.get('bot_goal_min_dist', 0)
            # Check distance to existing goals
            if goal_min > 0 and any(self._manhattan_distance(pos_dict, g) < goal_min for g in self.goals): return False
            # Check distance to existing bot starts
            if bot_min > 0 and any(self._manhattan_distance(pos_dict, sp) < bot_min for sp in self.start_positions): return False
            # Check distance to existing goals (for bot placement)
            if bot_goal_min > 0 and any(self._manhattan_distance(pos_dict, g) < bot_goal_min for g in self.goals): return False
            return True # All checks passed

        # Place Goals first as they influence bot placement
        min_goal_dist = max(2, int(self.size * self.config_factors['goal_dist']))
        attempts = 0
        goal_id_counter = 0
        while len(self.goals) < self.num_goals and attempts < max_placement_attempts:
            attempts += 1
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if is_valid_placement(pos, occupied, {'goal_min_dist': min_goal_dist}):
                goal = {'x': pos[0], 'y': pos[1], 'id': f'G{goal_id_counter}'}
                self.goals.append(goal)
                occupied.add(pos)
                goal_id_counter += 1
        if len(self.goals) < self.num_goals:
            print(f"Warning: Placed only {len(self.goals)}/{self.num_goals} goals after {attempts} attempts.")

        # Place Bots
        min_bot_dist = max(3, int(self.size * self.config_factors['bot_dist']))
        min_bot_goal_dist = max(3, int(self.size * self.config_factors['bot_goal_dist']))
        attempts = 0
        placed_bots = 0
        while placed_bots < total_bots and attempts < max_placement_attempts:
            attempts += 1
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if is_valid_placement(pos, occupied, {'bot_min_dist': min_bot_dist, 'bot_goal_min_dist': min_bot_goal_dist}):
                # Determine bot type based on counts
                bot_type = 'Hardcoded' if placed_bots < self.num_hc_bots else 'Learning'
                # Determine bot number within its type
                bot_num = placed_bots if bot_type == 'Hardcoded' else placed_bots - self.num_hc_bots
                bot_id = f'{bot_type[0]}{bot_num}' # H0, H1, ..., L0, L1, ...

                self.start_positions.append({'x': pos[0], 'y': pos[1], 'type': bot_type, 'id': bot_id})
                occupied.add(pos)
                placed_bots += 1
        if placed_bots < total_bots:
            print(f"CRITICAL Warning: Placed only {placed_bots}/{total_bots} bots after {attempts} attempts. Simulation will run with fewer bots.")

        # Place Obstacles last, avoiding goal and bot start positions
        num_obstacles_to_place = random.randint(
            int(total_cells * self.min_obstacles_factor),
            int(total_cells * self.max_obstacles_factor)
        ) if total_cells > 0 else 0
        attempts = 0
        placed_obstacles = 0
        while placed_obstacles < num_obstacles_to_place and attempts < max_placement_attempts:
             attempts += 1
             pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
             # Ensure obstacle doesn't block initial goals or bot starts
             if is_valid_placement(pos, occupied): # Check against already occupied (goals, bots, other obstacles)
                 self.obstacles.add(pos)
                 occupied.add(pos) # Add to occupied for future checks
                 placed_obstacles += 1
        # No warning needed if fewer obstacles placed, it just makes the map easier

        print(f"Environment randomized. Placed {len(self.goals)} goals, {len(self.start_positions)} bots, {len(self.obstacles)} obstacles.")


    def reset_round_state(self):
        self.claimed_goals.clear()
        # Reset goals from the initially generated set
        self.goals = [{'x': g['x'], 'y': g['y'], 'id': g['id']} for g in self._initial_goals]
        print(f"Round state reset. {len(self.goals)} goals active.")

    def is_valid(self, pos):
        if not pos or 'x' not in pos or 'y' not in pos: return False
        x, y = pos['x'], pos['y']
        # Check grid bounds and if the cell is an obstacle
        return 0 <= x < self.size and 0 <= y < self.size and (x, y) not in self.obstacles

    # --- Pathfinding Helper (Simple BFS for Hardcoded Bot) ---
    def find_path(self, start_pos, goal_pos, all_bots_dict=None):
        """ Finds a path using BFS, optionally avoiding other bots """
        if not self.is_valid(start_pos) or not self.is_valid(goal_pos): return None
        start_tuple = (start_pos['x'], start_pos['y'])
        goal_tuple = (goal_pos['x'], goal_pos['y'])
        if start_tuple == goal_tuple: return []

        # Create a set of temporary obstacle locations including other bots
        current_obstacles = self.obstacles.copy()
        if all_bots_dict:
            for bot_id, bot_state in all_bots_dict.items():
                # Don't avoid self
                if bot_state['pos']['x'] == start_pos['x'] and bot_state['pos']['y'] == start_pos['y']:
                    continue
                current_obstacles.add((bot_state['pos']['x'], bot_state['pos']['y']))

        queue = deque([(start_tuple, [])]) # Store (position, path_list)
        visited = {start_tuple}
        deltas = [(0, -1, 0), (-1, 0, 1), (1, 0, 2), (0, 1, 3)] # dy, dx, action_index

        while queue:
            current_pos_tuple, path = queue.popleft()
            if current_pos_tuple == goal_tuple:
                return path # Return list of actions

            for dy, dx, action in deltas:
                next_x, next_y = current_pos_tuple[0] + dx, current_pos_tuple[1] + dy
                next_pos_tuple = (next_x, next_y)
                if (0 <= next_x < self.size and
                    0 <= next_y < self.size and
                    next_pos_tuple not in current_obstacles and # Check combined obstacles
                    next_pos_tuple not in visited):
                    visited.add(next_pos_tuple)
                    new_path = path + [action]
                    queue.append((next_pos_tuple, new_path))
        return None # No path found


    def get_sensory_data(self, acting_bot, all_bots_dict, visibility_range):
        bot_pos = acting_bot['pos']
        vis_range = max(1, int(visibility_range))
        senses = {
            'wallDistance': min(bot_pos['x'], bot_pos['y'], self.size - 1 - bot_pos['x'], self.size - 1 - bot_pos['y']),
            'nearestVisibleGoalDist': vis_range + 1,
            'numVisibleGoals': 0,
            'nearestOpponentDist': vis_range + 1,
            'opponentIsFrozen': False,
            'opponentType': 'None', # Hardcoded, Learning, Player
            'isFrozen': acting_bot['freezeTimer'] > 0,
            '_visibleGoals': [], # List of {'x', 'y', 'id', 'dist'}
            '_nearestOpponent': None # *Simplified* opponent state if visible and nearest
        }

        # Find visible goals
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals:
                 dist = self._manhattan_distance(bot_pos, goal)
                 if dist <= vis_range:
                     senses['numVisibleGoals'] += 1
                     senses['_visibleGoals'].append({'x': goal['x'], 'y': goal['y'], 'id': goal['id'], 'dist': dist})
                     senses['nearestVisibleGoalDist'] = min(senses['nearestVisibleGoalDist'], dist)

        # Find nearest visible opponent
        nearest_opponent_obj = None
        min_opp_dist = vis_range + 1
        for opp_id, opponent_bot in all_bots_dict.items():
             if opp_id == acting_bot['id']: continue # Skip self
             dist = self._manhattan_distance(bot_pos, opponent_bot['pos'])
             if dist <= vis_range:
                 if dist < min_opp_dist:
                     min_opp_dist = dist
                     nearest_opponent_obj = opponent_bot # Store the actual bot state

        senses['nearestOpponentDist'] = min_opp_dist
        if nearest_opponent_obj:
            senses['opponentIsFrozen'] = nearest_opponent_obj['freezeTimer'] > 0
            is_player_controlled = nearest_opponent_obj.get('is_player_controlled', False)
            senses['opponentType'] = 'Player' if is_player_controlled else nearest_opponent_obj['type']

            # *** FIX for RecursionError ***
            # Create a *new dictionary* with only essential, serializable info
            # Avoid copying memory_state, policy_head, senses, etc.
            simplified_opponent_state = {
                'id': nearest_opponent_obj['id'],
                'type': nearest_opponent_obj['type'],
                'pos': copy.copy(nearest_opponent_obj['pos']), # Shallow copy is fine for pos dict
                'freezeTimer': nearest_opponent_obj['freezeTimer'],
                'is_player_controlled': is_player_controlled
                # Add other simple fields if needed by bot logic, but avoid complex objects
            }
            senses['_nearestOpponent'] = simplified_opponent_state
            # *** END FIX ***

        # Sort visible goals by distance
        senses['_visibleGoals'].sort(key=lambda g: g['dist'])

        # Cap distances at vis_range + 1 if nothing is seen
        for key in ['wallDistance', 'nearestVisibleGoalDist', 'nearestOpponentDist']:
            senses[key] = min(senses[key], vis_range + 1)

        return senses

    def perform_move_action(self, bot_pos, action_index):
        next_pos = bot_pos.copy()
        # 0:Up, 1:Left, 2:Right, 3:Down
        delta = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        if 0 <= action_index <= 3:
            dx, dy = delta[action_index]
            next_pos['x'] += dx
            next_pos['y'] += dy
        return next_pos # Returns the intended next position

    def get_adjacent_unclaimed_goal(self, bot_pos):
        for goal in self.goals:
            if goal['id'] not in self.claimed_goals:
                # Check if goal is exactly 1 step away (Manhattan distance)
                if self._manhattan_distance(bot_pos, goal) == 1:
                    return goal # Return the goal dictionary
        return None # No adjacent, unclaimed goal found

    def claim_goal(self, goal_id, bot_id):
        if goal_id in self.claimed_goals:
            # print(f"Attempt to claim already claimed goal {goal_id} by {bot_id}") # Debug
            return False # Goal already claimed
        # Check if the goal exists in the current list of goals
        goal_exists = any(g['id'] == goal_id for g in self.goals)
        if goal_exists:
            self.claimed_goals.add(goal_id)
            # print(f"Goal {goal_id} claimed by {bot_id}") # Debug
            return True # Successfully claimed
        else:
            # print(f"Attempt to claim non-existent/reset goal {goal_id} by {bot_id}") # Debug
            return False # Goal doesn't exist (e.g., after reset)

    def are_all_goals_claimed(self):
         # Check if the number of claimed goals equals or exceeds the number of *initial* goals
         return len(self._initial_goals) > 0 and len(self.claimed_goals) >= len(self._initial_goals)


    def get_state(self):
        # Return only *active* goals in the environment state for the frontend
        active_goals = [g for g in self.goals if g['id'] not in self.claimed_goals]
        return {
            'size': self.size,
            'goals': active_goals,
            'obstacles': list(self.obstacles), # Convert set to list for JSON
            'claimedGoals': list(self.claimed_goals) # Send claimed IDs too
        }

# ================================================================
# --- Bot Logic (Including Titans Learning & Player Takeover) ---
# ================================================================

# --- Hardcoded Bot Logic (Improved Stuck Handling v2) ---
def get_hardcoded_action(bot_state, senses, env, all_bots_dict):
    bot_id, pos = bot_state['id'], bot_state['pos']

    # Ensure necessary keys exist
    bot_state.setdefault('stuckCounter', 0)
    bot_state.setdefault('currentPath', None)
    bot_state.setdefault('lastMoveAttempt', -1)
    bot_state.setdefault('targetGoalId', None) # Track target goal

    if bot_state['freezeTimer'] > 0:
        bot_state['stuckCounter'] = 0
        bot_state['currentPath'] = None
        bot_state['targetGoalId'] = None
        return -1 # Frozen

    # --- Immediate Actions ---
    adjacent_goal = env.get_adjacent_unclaimed_goal(pos)
    if adjacent_goal:
        bot_state['stuckCounter'] = 0
        bot_state['currentPath'] = None
        bot_state['targetGoalId'] = None
        bot_state['lastMoveAttempt'] = 5
        return 5 # Action: Claim Goal

    nearest_opponent = senses.get('_nearestOpponent')
    if nearest_opponent and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen'):
        bot_state['stuckCounter'] = 0
        bot_state['currentPath'] = None
        bot_state['targetGoalId'] = None
        bot_state['lastMoveAttempt'] = 4
        return 4 # Action: Punch

    # --- Path Following ---
    current_path = bot_state.get('currentPath')
    if current_path:
        next_action = current_path[0] # Peek at next action
        intended_pos = env.perform_move_action(pos, next_action)

        # Check if the intended position is valid AND not occupied by another bot
        is_pos_valid = env.is_valid(intended_pos)
        is_pos_occupied_by_other = False
        if is_pos_valid:
            for other_id, other_bot in all_bots_dict.items():
                if other_id != bot_id and other_bot['pos'] == intended_pos:
                    is_pos_occupied_by_other = True
                    break

        if is_pos_valid and not is_pos_occupied_by_other:
            # Path is clear, take the step
            bot_state['lastMoveAttempt'] = current_path.pop(0)
            bot_state['stuckCounter'] = 0
            if not current_path: # Path finished
                bot_state['targetGoalId'] = None
            return bot_state['lastMoveAttempt']
        else:
            # Path blocked (wall, obstacle, or bot)
            # print(f"D: {bot_id} path blocked at {intended_pos}. Reason: {'Invalid' if not is_pos_valid else 'Occupied'}. Recalculating.")
            bot_state['currentPath'] = None # Clear blocked path
            bot_state['stuckCounter'] += 1 # Increment stuck counter
            # Fall through to recalculate or wander

    # --- Path Calculation / Recalculation ---
    visible_goals = senses.get('_visibleGoals', [])
    target_goal = None

    # Try to stick to the previous target if still visible and unclaimed
    if bot_state['targetGoalId']:
        potential_target = next((g for g in visible_goals if g['id'] == bot_state['targetGoalId']), None)
        if potential_target:
            target_goal = potential_target
        else:
            bot_state['targetGoalId'] = None # Target no longer visible/valid

    # If no valid previous target, pick the nearest visible one
    if not target_goal and visible_goals:
        target_goal = visible_goals[0]
        bot_state['targetGoalId'] = target_goal['id']

    # If we have a target goal, try to pathfind
    if target_goal:
        # print(f"D: {bot_id} targeting goal {target_goal['id']}. Finding path...")
        path_to_goal = env.find_path(pos, target_goal, all_bots_dict) # Avoid other bots

        if path_to_goal:
            # print(f"D: {bot_id} found path to {target_goal['id']}: {path_to_goal}")
            bot_state['currentPath'] = path_to_goal
            if bot_state['currentPath']: # Path might be empty if already at goal
                next_action = bot_state['currentPath'].pop(0)
                bot_state['lastMoveAttempt'] = next_action
                bot_state['stuckCounter'] = 0
                if not bot_state['currentPath']: bot_state['targetGoalId'] = None # Path finished
                return next_action
            else: # Path is empty, should have claimed
                 bot_state['targetGoalId'] = None
                 bot_state['currentPath'] = None
                 # Fall through to wander if claim fails somehow
        else:
            # print(f"W: {bot_id} sees goal {target_goal['id']} but cannot find path (maybe blocked by bots).")
            bot_state['targetGoalId'] = None # Clear target if no path found
            bot_state['stuckCounter'] += 1
            # Fall through to random move

    else:
        # No goals visible or no path found, explore randomly
        bot_state['stuckCounter'] += 1
        bot_state['targetGoalId'] = None

    # --- Fallback: Random Move (Improved Stuck Avoidance v2) ---
    valid_moves = []
    for action_idx in range(4):
        next_p = env.perform_move_action(pos, action_idx)
        if env.is_valid(next_p):
            # Check if occupied by another bot
            occupied = any(bid != bot_id and b['pos'] == next_p for bid, b in all_bots_dict.items())
            if not occupied:
                valid_moves.append(action_idx)

    if not valid_moves:
        bot_state['lastMoveAttempt'] = -1
        # print(f"D: {bot_id} trapped, no valid moves.")
        return -1 # Trapped

    last_move = bot_state.get('lastMoveAttempt', -1)
    reverse_action = -1
    if 0 <= last_move <= 3:
        reverse_map = {0: 3, 1: 2, 2: 1, 3: 0}
        reverse_action = reverse_map[last_move]

    non_reverse_moves = [m for m in valid_moves if m != reverse_action]

    chosen_move = -1
    # Prioritize non-reversing moves unless stuck for a while
    if non_reverse_moves and bot_state['stuckCounter'] < 3:
        chosen_move = random.choice(non_reverse_moves)
    elif valid_moves: # Allow reversing if stuck or only option
        chosen_move = random.choice(valid_moves)
        # if bot_state['stuckCounter'] >= 3: print(f"D: {bot_id} stuck ({bot_state['stuckCounter']}), allowing reverse: {chosen_move}")

    bot_state['lastMoveAttempt'] = chosen_move
    if chosen_move != -1:
        # Only reset stuck counter if the move was *different* from the last attempt
        # This helps break out of back-and-forth oscillations
        if chosen_move != bot_state.get('lastMoveAttempt'):
             bot_state['stuckCounter'] = 0
        else: # If repeating the same move, might still be stuck
             bot_state['stuckCounter'] = max(0, bot_state['stuckCounter'] - 1) # Decrease slowly if repeating
    bot_state['currentPath'] = None # Clear path as we are wandering
    return chosen_move


# --- Learning Bot Input Encoding ---
def _get_input_tensor_for_bot(bot_state, senses):
    """ Encodes bot's senses and last action into a tensor for the NeuralMemory """
    dim = current_config['LEARNING_BOT_DIM']
    vis_range = current_config['VISIBILITY_RANGE']
    num_actions = current_config['NUM_ACTIONS']

    if dim <= 0:
        raise ValueError("LEARNING_BOT_DIM must be positive.")

    features = []

    # Normalize distance features (0=close, 1=far/unseen)
    def norm_dist(d):
        d_num = float(d) if isinstance(d, (int, float)) else float('inf')
        # Normalize relative to visibility range
        return min(1.0, max(0.0, d_num / vis_range)) if vis_range > 0 else 0.0

    # Core Senses
    features.append(norm_dist(senses.get('wallDistance', vis_range + 1)))
    features.append(norm_dist(senses.get('nearestVisibleGoalDist', vis_range + 1)))
    # Normalize number of goals (e.g., capped at 5)
    features.append(min(1.0, max(0.0, senses.get('numVisibleGoals', 0) / 5.0)))
    features.append(norm_dist(senses.get('nearestOpponentDist', vis_range + 1)))
    features.append(1.0 if senses.get('opponentIsFrozen', False) else 0.0)
    features.append(1.0 if senses.get('isFrozen', False) else 0.0)

    # Opponent Type (One-hot encoding)
    opp_type = senses.get('opponentType', 'None')
    opp_types = ['Hardcoded', 'Learning', 'Player', 'None'] # Consistent order
    opp_type_enc = [0.0] * len(opp_types)
    if opp_type in opp_types:
        opp_type_enc[opp_types.index(opp_type)] = 1.0
    features.extend(opp_type_enc)

    # Last Action (One-hot encoding, -1 maps to all zeros)
    last_action = bot_state.get('lastAction', -1)
    action_enc = [0.0] * num_actions
    if 0 <= last_action < num_actions:
        action_enc[last_action] = 1.0
    features.extend(action_enc)

    # Padding / Truncating to match LEARNING_BOT_DIM
    current_len = len(features)
    if current_len < dim:
        # Pad with zeros
        features.extend([0.0] * (dim - current_len))
    elif current_len > dim:
        # This should ideally not happen if DIM is set large enough
        print(f"Warning: Feature vector length ({current_len}) > DIM ({dim}). Truncating features for bot {bot_state.get('id','?')}.")
        features = features[:dim]

    # Create tensor on the correct device
    input_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # Shape [1, 1, dim]

    # Final shape check
    if input_tensor.shape != (1, 1, dim):
        raise ValueError(f"Internal Error: Final input tensor shape is {input_tensor.shape}, expected [1, 1, {dim}] for bot {bot_state.get('id','?')}")

    return input_tensor


# --- Learning Bot Action Selection (Uses NeuralMemory & Player Control) ---
def get_learning_action(bot_state, senses, env, all_bots_dict, direct_player_action):
    """ Determines the action for a learning bot, handling AI, player target, and direct player actions """
    bot_id = bot_state['id']

    # Handle Frozen State
    if bot_state['freezeTimer'] > 0:
        bot_state['mode'] = "Frozen"
        # Clear any pending player input if frozen
        bot_state['target_coordinate'] = None
        return -1 # Cannot act

    chosen_action = -1 # Default to idle

    # --- Player Control Logic ---
    if bot_state.get('is_player_controlled', False):
        # Priority 1: Direct action from player (e.g., mobile button)
        if direct_player_action is not None and 0 <= direct_player_action < current_config['NUM_ACTIONS']:
            chosen_action = direct_player_action
            bot_state['mode'] = f"Player (Direct: {chosen_action})"
            # Clear target if a direct action is given? Maybe not, let target persist.
            # bot_state['target_coordinate'] = None
        # Priority 2: Follow target coordinate if set
        elif bot_state.get('target_coordinate'):
            target = bot_state['target_coordinate']
            current_pos = bot_state['pos']
            dx = target['x'] - current_pos['x']
            dy = target['y'] - current_pos['y']
            dist = abs(dx) + abs(dy)

            if dist == 0:
                # Reached target, clear target and idle this step
                bot_state['target_coordinate'] = None
                chosen_action = -1
                bot_state['mode'] = "Player (Target Reached)"
            else:
                # Try to interact if adjacent to target
                if dist == 1:
                    # Check for opponent to punch at target
                    opponent_at_target = next((b for bid, b in all_bots_dict.items() if bid != bot_id and b['pos'] == target and b['freezeTimer'] <= 0), None)
                    if opponent_at_target:
                        chosen_action = 4 # Punch
                        bot_state['mode'] = f"Player (Target Punch {opponent_at_target['id']})"
                    else:
                        # Check for goal to claim at target
                        goal_at_target = next((g for g in env.goals if g['id'] not in env.claimed_goals and g['x'] == target['x'] and g['y'] == target['y']), None)
                        if goal_at_target:
                            chosen_action = 5 # Claim
                            bot_state['mode'] = f"Player (Target Claim {goal_at_target['id']})"

                # If no interaction, move towards target
                if chosen_action == -1:
                    move_options = []
                    # Prioritize moves that reduce distance
                    if abs(dx) >= abs(dy): # Move horizontally first
                        move_options.append(2 if dx > 0 else 1) # Right or Left
                        if dy != 0: move_options.append(3 if dy > 0 else 0) # Then Down or Up
                    else: # Move vertically first
                        move_options.append(3 if dy > 0 else 0) # Down or Up
                        if dx != 0: move_options.append(2 if dx > 0 else 1) # Then Right or Left
                    # Add remaining moves as lower priority
                    for m in range(4):
                        if m not in move_options: move_options.append(m)

                    # Find the first valid move from the prioritized list
                    valid_move_found = False
                    for move in move_options:
                        intended_pos = env.perform_move_action(current_pos, move)
                        if env.is_valid(intended_pos):
                             # Check if occupied by another bot
                             occupied = any(bid != bot_id and b['pos'] == intended_pos for bid, b in all_bots_dict.items())
                             if not occupied:
                                 chosen_action = move
                                 valid_move_found = True
                                 break
                    if not valid_move_found:
                         chosen_action = -1 # Blocked, idle this step
                         bot_state['mode'] = "Player (Target Blocked)"
                    else:
                         bot_state['mode'] = f"Player (Target Move {chosen_action})"

        # Priority 3: Idle if no direct action and no target
        else:
            chosen_action = -1
            bot_state['mode'] = "Player (Idle)"

    # --- AI Control Logic ---
    else:
        # Ensure NN manager and bot components are ready
        if not neural_memory_manager:
             print(f"Error: Neural Memory Manager not ready for AI bot {bot_id}. Bot cannot act.")
             bot_state['mode'] = "Error (No NN Manager)"
             return -1

        memory_state = bot_state.get('memory_state')
        policy_head = bot_state.get('policy_head')

        if not memory_state or not policy_head:
            print(f"Error: Learning components missing for AI bot {bot_id}. Bot cannot act.")
            bot_state['mode'] = "Error (Missing Components)"
            return -1

        # --- Action Selection (Exploration vs Exploitation) ---
        explore_roll = random.random()
        exploration_threshold = current_config['LEARNING_BOT_BASE_EXPLORATION_RATE']
        is_exploring = explore_roll < exploration_threshold

        if is_exploring:
            # Exploration Mode
            rule_explore_roll = random.random()
            if rule_explore_roll < current_config['LEARNING_BOT_RULE_EXPLORE_PERCENT']:
                bot_state['mode'] = "Explore (Rule)"
                # Ensure HC state keys exist before calling
                bot_state.setdefault('stuckCounter', 0)
                bot_state.setdefault('currentPath', None)
                bot_state.setdefault('lastMoveAttempt', -1)
                bot_state.setdefault('targetGoalId', None)
                chosen_action = get_hardcoded_action(bot_state, senses, environment, all_bots_dict) # Use HC logic
            else:
                bot_state['mode'] = "Explore (Random)"
                chosen_action = random.choice(list(range(current_config['NUM_ACTIONS'])))

            # --- Validate Random/Rule Exploration Action ---
            temp_pos = bot_state['pos'].copy()
            # Check Punch validity
            if chosen_action == 4 and not (senses.get('_nearestOpponent') and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen')):
                chosen_action = random.randint(0, 3) # Fallback to random move
            # Check Claim validity
            elif chosen_action == 5 and not env.get_adjacent_unclaimed_goal(temp_pos):
                chosen_action = random.randint(0, 3) # Fallback to random move

            # Check Move validity (if action is still a move)
            if 0 <= chosen_action <= 3:
                 intended_pos = env.perform_move_action(temp_pos, chosen_action)
                 if not env.is_valid(intended_pos) or any(bid != bot_id and b['pos'] == intended_pos for bid, b in all_bots_dict.items()):
                     valid_moves = [m for m in range(4) if env.is_valid(env.perform_move_action(temp_pos, m)) and not any(bid != bot_id and b['pos'] == env.perform_move_action(temp_pos, m) for bid, b in all_bots_dict.items())]
                     chosen_action = random.choice(valid_moves) if valid_moves else -1 # Fallback

        else:
            # Exploitation Mode: Use policy head on retrieved memory output
            bot_state['mode'] = "Exploit"
            try:
                # --- Prepare Input Tensor ---
                input_tensor = _get_input_tensor_for_bot(bot_state, senses)

                # --- Process Memory & Get Prediction ---
                # Retrieve based on current state *before* update
                # We need the retrieved value to feed into the policy head
                retrieved_tensor, _, _ = neural_memory_manager.forward_step(
                    input_tensor, memory_state, detach_next_state=False # Don't need next state here, just retrieval
                )

                # --- Get Action from Policy Head ---
                policy_head.eval()
                policy_head = policy_head.to(device)
                retrieved_tensor_on_device = retrieved_tensor.to(device)

                with torch.no_grad():
                    action_logits = policy_head(retrieved_tensor_on_device.squeeze(0))
                    action_probs = F.softmax(action_logits, dim=-1)

                # --- Mask Invalid Actions ---
                mask = torch.ones_like(action_probs, dtype=torch.bool, device=device)
                # Check Punch
                if not (senses.get('_nearestOpponent') and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen')): mask[0, 4] = False
                # Check Claim
                if not env.get_adjacent_unclaimed_goal(bot_state['pos']): mask[0, 5] = False
                # Check Moves
                for move_idx in range(4):
                    intended_pos = env.perform_move_action(bot_state['pos'], move_idx)
                    if not env.is_valid(intended_pos) or any(bid != bot_id and b['pos'] == intended_pos for bid, b in all_bots_dict.items()):
                        mask[0, move_idx] = False

                masked_probs = action_probs * mask.float()

                if masked_probs.sum().item() < 1e-6:
                     print(f"W: AI Bot {bot_id} has no valid actions from policy, choosing random valid.")
                     valid_actions = [a for a in range(current_config['NUM_ACTIONS']) if mask[0, a].item()]
                     chosen_action = random.choice(valid_actions) if valid_actions else -1
                     bot_state['mode'] += " (Fallback)"
                else:
                    # Normalize masked probabilities before sampling
                    masked_probs /= masked_probs.sum()
                    chosen_action = torch.multinomial(masked_probs, num_samples=1).item()
                    bot_state['mode'] += f" (Predict {chosen_action})"

            except Exception as e:
                print(f"Error: Exploitation phase failed for AI bot {bot_id}: {e}")
                traceback.print_exc()
                bot_state['mode'] = "Error (Exploitation)"
                # Fallback to simple random valid move
                valid_moves = [m for m in range(4) if env.is_valid(env.perform_move_action(bot_state['pos'], m)) and not any(bid != bot_id and b['pos'] == env.perform_move_action(bot_state['pos'], m) for bid, b in all_bots_dict.items())]
                chosen_action = random.choice(valid_moves) if valid_moves else -1

    # --- Final Action Validation (Safety Check) ---
    final_action = chosen_action
    temp_pos = bot_state['pos'].copy()
    if final_action == 4 and not (senses.get('_nearestOpponent') and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen')):
        final_action = -1 # Invalid punch
    elif final_action == 5 and not env.get_adjacent_unclaimed_goal(temp_pos):
        final_action = -1 # Invalid claim
    elif 0 <= final_action <= 3:
        intended_pos = env.perform_move_action(temp_pos, final_action)
        if not env.is_valid(intended_pos) or any(bid != bot_id and b['pos'] == intended_pos for bid, b in all_bots_dict.items()):
            final_action = -1 # Invalid move

    # --- Update Memory (Always happens for Learning bots, using the final chosen action) ---
    if bot_state['type'] == 'Learning': # Only update memory for learning bots
        try:
            # Use the *final_action* decided above to encode the input for the memory update
            bot_state_for_update = bot_state.copy()
            bot_state_for_update['lastAction'] = final_action # Use the action actually taken (or -1 if idle/invalid)
            input_tensor_for_update = _get_input_tensor_for_bot(bot_state_for_update, senses)

            # Perform the memory update step
            _, next_memory_state, loss_value = neural_memory_manager.forward_step(
                input_tensor_for_update,
                bot_state['memory_state'],
                detach_next_state=True # Store detached state back
            )
            bot_state['memory_state'] = next_memory_state
            # Update anomaly using EMA
            bot_state['last_anomaly_proxy'] = loss_value * 0.1 + bot_state.get('last_anomaly_proxy', 0.0) * 0.9
        except Exception as e:
            print(f"Error: NeuralMemory update failed for bot {bot_id} (Action: {final_action}): {e}")
            traceback.print_exc()
            # Don't change mode here, error is internal to learning

    # Record the final action taken (for next step's input encoding)
    bot_state['lastAction'] = final_action
    return final_action


# ================================================================
# --- Simulation Setup & Control (Modified for NN Manager & Player Takeover) ---
# ================================================================

def create_learning_bot_instance(bot_id, start_pos):
    """ Creates the state dictionary for a new learning bot using the manager """
    if not neural_memory_manager:
         print(f"FATAL: Cannot create learning bot {bot_id}, NN Manager not ready.")
         raise RuntimeError("Neural Memory Manager is not initialized.")

    print(f"Creating Learning Bot {bot_id} using NN Manager (DIM={neural_memory_manager.dim}) on {device}")
    initial_mem_state = neural_memory_manager.get_initial_state() # Get state from manager

    # Create a dedicated policy head instance for this bot, on target_device
    policy_head = nn.Linear(neural_memory_manager.dim, current_config['NUM_ACTIONS']).to(device)
    nn.init.xavier_uniform_(policy_head.weight)
    if policy_head.bias is not None: nn.init.zeros_(policy_head.bias)

    return {
        'id': bot_id, 'type': 'Learning', 'pos': start_pos.copy(),
        'steps': 0, 'goalsReachedThisRound': 0, 'goalsReachedTotal': 0,
        'freezeTimer': 0, 'lastAction': -1, 'mode': 'Init', 'senses': {},
        'memory_state': initial_mem_state, # NeuralMemState tuple (contains weights, optim state)
        'policy_head': policy_head, # Instance of policy head nn.Module
        'last_anomaly_proxy': 0.0, # EMA of anomaly/loss
        'is_player_controlled': False, # Flag for player takeover
        'target_coordinate': None,     # Player target {x: int, y: int} or None
        'original_bot_id': bot_id,      # Store its own ID initially
        # --- Add keys needed by hardcoded logic (for exploration) ---
        'stuckCounter': 0,
        'lastMoveAttempt': -1,
        'currentPath': None,
        'targetGoalId': None
    }

# Hardcoded instance creation remains the same
def create_hardcoded_bot_instance(bot_id, start_pos):
     return {'id': bot_id, 'type': 'Hardcoded', 'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0, 'goalsReachedTotal': 0, 'freezeTimer': 0, 'lastAction': -1, 'mode': 'Init', 'senses': {}, 'stuckCounter': 0, 'lastMoveAttempt': -1, 'currentPath': None, 'targetGoalId': None}


def setup_simulation(full_reset=False):
    global environment, bots, round_number, stats, current_config, players, player_direct_actions
    print(f"--- Setting up Simulation (Full Reset: {full_reset}) ---")

    if full_reset:
        print("Performing full reset...")
        round_number = 0
        stats = {'hc_total_goals': 0, 'learning_total_goals': 0}
        # Update NN manager FIRST if learning params changed
        update_neural_memory_manager()
        if not neural_memory_manager: return False # Stop if manager fails

        print("Clearing existing bot states...")
        # Clean up PyTorch modules before clearing dict
        for bot_id, bot_state in list(bots.items()):
             if 'policy_head' in bot_state and isinstance(bot_state['policy_head'], nn.Module):
                 del bot_state['policy_head'] # Let Python GC handle it
        bots.clear()
        players.clear() # Clear player associations
        player_direct_actions.clear() # Clear pending actions
        environment = None
        if device.type == 'cuda':
             print("Clearing CUDA cache...")
             torch.cuda.empty_cache()
    else: # Not a full reset, just a new round
        round_number += 1
        if environment:
            environment.reset_round_state() # Reset claimed goals etc.
        else:
            # If environment doesn't exist, we must do a full setup
            print("No environment found for round reset, forcing full reset.")
            full_reset = True # Force the full reset logic below
        # Clear pending direct actions for the new round
        player_direct_actions.clear()

    env_needs_recreate = (environment is None or full_reset)

    if env_needs_recreate:
         # These parameters require environment recreation if changed
         env_defining_params = ['GRID_SIZE', 'NUM_GOALS', 'NUM_HC_BOTS', 'NUM_LEARNING_BOTS']
         # Check if these params changed IF it wasn't a forced full_reset
         # If it was a forced full_reset, we recreate regardless
         needs_recreate_due_to_params = full_reset or any(current_config[k] != getattr(environment, k.lower(), None) for k in env_defining_params if hasattr(environment, k.lower()))

         if needs_recreate_due_to_params:
             print(f"Recreating environment (Size: {current_config['GRID_SIZE']}, Goals: {current_config['NUM_GOALS']}, HC: {current_config['NUM_HC_BOTS']}, Lrn: {current_config['NUM_LEARNING_BOTS']})...")
             try:
                 obstacle_range = (current_config['OBSTACLES_FACTOR_MIN'], current_config['OBSTACLES_FACTOR_MAX'])
                 dist_factors = {
                     'MIN_GOAL_START_DISTANCE_FACTOR': current_config.get('MIN_GOAL_START_DISTANCE_FACTOR', 0.15),
                     'MIN_BOT_START_DISTANCE_FACTOR': current_config.get('MIN_BOT_START_DISTANCE_FACTOR', 0.25),
                     'MIN_BOT_GOAL_DISTANCE_FACTOR': current_config.get('MIN_BOT_GOAL_DISTANCE_FACTOR', 0.15)
                 }
                 environment = GridEnvironment(
                     current_config['GRID_SIZE'], current_config['NUM_GOALS'], obstacle_range,
                     current_config['NUM_HC_BOTS'], current_config['NUM_LEARNING_BOTS'], dist_factors
                 )
                 # If env recreated, bots must be recreated or repositioned
                 full_reset = True # Treat it as a full reset for bot handling
             except Exception as e:
                 print(f"FATAL: Environment creation failed: {e}")
                 traceback.print_exc()
                 return False
         elif not full_reset and environment: # Env exists, params didn't force recreate, just reset round state
              environment.reset_round_state()

    # Create/Reset Bot States
    new_bots = {}
    required_hc = current_config['NUM_HC_BOTS']
    required_ln = current_config['NUM_LEARNING_BOTS']
    required_total = required_hc + required_ln
    bot_starts = environment.start_positions if environment else []

    # Verify start positions match required bots
    if len(bot_starts) != required_total:
        print(f"Error: Environment start positions ({len(bot_starts)}) != required bots ({required_total}).")
        # Attempt to run with available starts, adjusting required counts
        available_total = len(bot_starts)
        available_hc = min(required_hc, available_total)
        available_ln = min(required_ln, available_total - available_hc)
        print(f"Attempting to run with adjusted counts: HC={available_hc}, LN={available_ln}")
        required_hc = available_hc
        required_ln = available_ln
        if len(bot_starts) < required_hc + required_ln:
             print("FATAL MISMATCH even after adjustment. Cannot setup simulation.")
             return False

    start_idx = 0
    try:
        # Hardcoded Bots
        for i in range(required_hc):
            start_pos_data = bot_starts[start_idx]
            bot_id = start_pos_data['id'] # Use ID from env generation
            start_pos = {'x': start_pos_data['x'], 'y': start_pos_data['y']}
            if full_reset or bot_id not in bots:
                new_bots[bot_id] = create_hardcoded_bot_instance(bot_id, start_pos)
            else: # Reset existing bot
                bots[bot_id].update({
                    'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0,
                    'freezeTimer': 0, 'lastAction': -1, 'mode': 'Reset',
                    'stuckCounter': 0, 'lastMoveAttempt': -1, 'currentPath': None, 'targetGoalId': None
                })
                new_bots[bot_id] = bots[bot_id] # Keep existing object
            start_idx += 1

        # Learning Bots
        for i in range(required_ln):
            start_pos_data = bot_starts[start_idx]
            bot_id = start_pos_data['id'] # Use ID from env generation
            start_pos = {'x': start_pos_data['x'], 'y': start_pos_data['y']}
            # Check if this bot is currently player controlled
            controlling_sid = None
            for sid, p_data in players.items():
                if p_data['original_bot_id'] == bot_id:
                    controlling_sid = sid
                    break

            if full_reset or bot_id not in bots:
                 if neural_memory_manager:
                    new_bots[bot_id] = create_learning_bot_instance(bot_id, start_pos)
                    # If a player was controlling this ID before reset, re-establish link
                    if controlling_sid:
                        new_bots[bot_id]['is_player_controlled'] = True
                        players[controlling_sid]['player_bot_id'] = bot_id # Update player dict just in case
                        print(f"Re-established player {controlling_sid} control over new bot instance {bot_id}")
                 else:
                    print(f"W: Cannot create new learning bot {bot_id}, NN Manager not ready.")
                    if controlling_sid: del players[controlling_sid] # Remove player if bot cannot be created
                    continue # Skip if manager failed
            else: # Reset existing learning bot
                 bots[bot_id].update({
                    'pos': start_pos.copy(), 'steps': 0, 'goalsReachedThisRound': 0,
                    'freezeTimer': 0, 'lastAction': -1, 'mode': 'Reset',
                    'last_anomaly_proxy': 0.0,
                    'target_coordinate': None, # Reset player target
                    # --- Reset keys needed by hardcoded logic ---
                    'stuckCounter': 0,
                    'lastMoveAttempt': -1,
                    'currentPath': None,
                    'targetGoalId': None
                 })
                 # Reset memory state ONLY (keep policy head weights)
                 if neural_memory_manager:
                    bots[bot_id]['memory_state'] = neural_memory_manager.get_initial_state()
                 else:
                    print(f"Warning: NN Manager not ready during reset for {bot_id}. Memory state not reset.")

                 # Preserve player control if applicable
                 if controlling_sid:
                     bots[bot_id]['is_player_controlled'] = True
                     bots[bot_id]['mode'] = "Player Reset"
                     players[controlling_sid]['player_bot_id'] = bot_id # Ensure player dict is correct
                 else:
                     bots[bot_id]['is_player_controlled'] = False # Ensure it's reset if no player

                 new_bots[bot_id] = bots[bot_id] # Keep existing object
            start_idx += 1

        # Update the main bots dictionary
        bots = new_bots

        # Remove any player associations if the corresponding bot no longer exists
        # (Should be handled above, but double-check)
        for sid, player_data in list(players.items()):
            if player_data['original_bot_id'] not in bots:
                print(f"Removing player SID {sid} as their bot {player_data['original_bot_id']} no longer exists after setup.")
                del players[sid]


    except IndexError:
        print(f"Error: Index out of bounds during bot setup. bot_starts: {len(bot_starts)}, start_idx: {start_idx}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Error: Bot creation/reset failed: {e}")
        traceback.print_exc()
        return False

    print(f"Setup complete for Round {round_number}. Active Bots: {list(bots.keys())}")
    return True


def get_game_state():
    """ Creates a JSON-serializable representation of the game state """
    serializable_bots = {}
    for bot_id, bot_state in bots.items():
        s_bot = {
            'id': bot_state['id'],
            'type': bot_state['type'],
            'pos': bot_state['pos'],
            'freezeTimer': bot_state['freezeTimer'],
            'mode': bot_state['mode'],
            'goals_round': bot_state.get('goalsReachedThisRound', 0),
            'is_player': bot_state.get('is_player_controlled', False), # Indicate if player controlled
            'target_coord': bot_state.get('target_coordinate') # Send target coordinate
        }
        if bot_state['type'] == 'Learning':
            s_bot['anomaly'] = round(bot_state.get('last_anomaly_proxy', 0.0), 5)
            # Add is_player_controlled flag specifically for learning bots in UI
            s_bot['is_player_controlled'] = bot_state.get('is_player_controlled', False)
        serializable_bots[bot_id] = s_bot

    # Send the FULL current config to the client
    # The client can decide what to display
    return {
        'environment': environment.get_state() if environment else None,
        'bots': serializable_bots,
        'round': round_number,
        'stats': stats,
        'config': current_config # Send the full config
    }


def simulation_step():
    """ Performs one step of the simulation for all active bots """
    global player_direct_actions
    if not environment or not bots:
        print("Warning: Simulation step skipped, environment or bots not ready.")
        return False # Indicate failure or end condition

    round_over = False
    max_steps_reached_for_all = True # Assume true initially
    bot_ids_this_step = list(bots.keys()) # Get IDs at start of step

    # Process direct actions received since last step
    current_direct_actions = player_direct_actions.copy()
    player_direct_actions.clear() # Clear buffer for next step

    for bot_id in bot_ids_this_step:
        if bot_id not in bots: continue # Bot might have been removed

        bot_state = bots[bot_id]

        # Check if bot has reached max steps for this round
        if bot_state['steps'] >= current_config['MAX_STEPS_PER_ROUND']:
            continue # Skip this bot
        else:
            max_steps_reached_for_all = False # At least one bot can still move

        action = -1 # Default action (idle/frozen)
        next_pos = bot_state['pos'].copy() # Calculate potential next position

        try:
            # 1. Get Sensory Data
            bot_state['senses'] = environment.get_sensory_data(bot_state, bots, current_config['VISIBILITY_RANGE'])

            # 2. Determine Action (based on type and state)
            if bot_state['freezeTimer'] > 0:
                action = -1
                bot_state['mode'] = "Frozen"
            elif bot_state['type'] == 'Hardcoded':
                action = get_hardcoded_action(bot_state, bot_state['senses'], environment, bots)
                # Mode is set within get_hardcoded_action
            elif bot_state['type'] == 'Learning':
                # Check if there's a direct action for this player-controlled bot
                direct_action = None
                if bot_state.get('is_player_controlled', False):
                    # Find the SID controlling this bot
                    controlling_sid = next((sid for sid, p_data in players.items() if p_data['player_bot_id'] == bot_id), None)
                    if controlling_sid and controlling_sid in current_direct_actions:
                        direct_action = current_direct_actions[controlling_sid]

                # Get action (handles AI, player target, and direct action override)
                action = get_learning_action(bot_state, bot_state['senses'], environment, bots, direct_action)
                # Mode is set within get_learning_action

            # Ensure action is a valid integer
            if not isinstance(action, int) or action < -1 or action >= current_config['NUM_ACTIONS']:
                 print(f"Warning: Invalid action type or value ({action}) received for bot {bot_id}. Setting to idle (-1).")
                 action = -1

            # Record the final action decided (even if -1)
            # This is now done inside get_learning_action for memory update consistency
            # bot_state['lastAction'] = action

            # 3. Execute Action
            if action == -1 or bot_state['freezeTimer'] > 0:
                pass # Bot is idle or frozen, position doesn't change
            elif 0 <= action <= 3: # Move Action
                intended_pos = environment.perform_move_action(bot_state['pos'], action)
                # Check validity AND if occupied by another bot
                occupied = any(bid != bot_id and b['pos'] == intended_pos for bid, b in bots.items())
                if environment.is_valid(intended_pos) and not occupied:
                    next_pos = intended_pos # Move is valid
                else:
                    # Hit wall, obstacle, or bot, position doesn't change
                    bot_state['mode'] += " (Blocked)" # Append info to mode
            elif action == 4: # Punch Action
                # Find adjacent, unfrozen target
                target_bot = next((
                    ob for ob_id, ob in bots.items()
                    if ob_id != bot_id and environment._manhattan_distance(bot_state['pos'], ob['pos']) == 1 and ob['freezeTimer'] <= 0
                ), None)
                if target_bot:
                    target_bot['freezeTimer'] = current_config['FREEZE_DURATION']
                    bot_state['mode'] += f" (Punch {target_bot['id']})"
                else:
                    bot_state['mode'] += " (Punch Miss)"
            elif action == 5: # Claim Goal Action
                adjacent_goal = environment.get_adjacent_unclaimed_goal(bot_state['pos'])
                if adjacent_goal:
                     # Attempt to claim the goal
                     if environment.claim_goal(adjacent_goal['id'], bot_id):
                          bot_state['goalsReachedThisRound'] += 1
                          bot_state['goalsReachedTotal'] += 1
                          # Update overall stats (learning includes player goals)
                          if bot_state['type'] == 'Hardcoded':
                               stats['hc_total_goals'] += 1
                          else: # Learning or Player
                               stats['learning_total_goals'] += 1

                          bot_state['mode'] += f" (Claim {adjacent_goal['id']})"
                          # Check if all goals are now claimed
                          if environment.are_all_goals_claimed():
                               round_over = True
                               print(f"--- Round {round_number} Over: All goals claimed! (Final goal {adjacent_goal['id']} by {bot_id}) ---")
                               break # Exit the bot loop immediately
                     else: # Claim failed (e.g., race condition, already claimed)
                          bot_state['mode'] += " (Claim Failed)"
                else: # No adjacent goal to claim
                     bot_state['mode'] += " (Claim Miss)"

            # 4. Update Bot State
            bot_state['pos'] = next_pos # Update position
            bot_state['steps'] += 1 # Increment step count

            # Decrement freeze timer if frozen
            if bot_state['freezeTimer'] > 0:
                bot_state['freezeTimer'] -= 1

        except Exception as e:
            print(f"Error processing bot {bot_id} during simulation step: {e}")
            traceback.print_exc()
            # Stop the simulation on critical error
            global simulation_running
            simulation_running = False
            socketio.emit('simulation_stopped', {'message': f'Error processing bot {bot_id}. Simulation stopped.'})
            return False # Indicate failure/stop

    # Check if round ended due to max steps after processing all bots
    if not round_over and max_steps_reached_for_all:
        round_over = True
        print(f"--- Round {round_number} Over: Max steps reached for all bots! ---")

    # Return 'not round_over' (True to continue, False to stop/reset)
    return not round_over


# --- Simulation Loop (using eventlet for async sleep) ---
# Changed from async def to def
def simulation_loop():
    global simulation_running, round_number, simulation_loop_task
    print("Simulation loop started.")
    loop_count = 0
    emit_interval_steps = 3 # Emit state every N steps for smoother updates
    last_emit_time = time.monotonic()
    min_emit_interval_time = 0.05 # Emit at least every 50ms if steps are slow

    while simulation_running:
        loop_start_time = time.monotonic()
        try:
            # Perform one step for all bots
            continue_round = simulation_step()

            # Check if simulation was stopped externally (e.g., by user or error)
            if not simulation_running:
                print("Simulation running flag is false, exiting loop.")
                break

            # If the round ended (either by goals or max steps)
            if not continue_round:
                 print(f"Round {round_number} ended. Setting up next round...")
                 # Setup the next round (or initial if first time)
                 if setup_simulation(full_reset=False):
                     loop_count = 0 # Reset step counter for emit logic
                     last_emit_time = time.monotonic()
                     emit_state() # Emit the state of the new round
                     print(f"Starting Round {round_number}...")
                 else:
                     # Failed to setup next round, stop simulation
                     print("Error: Failed next round setup. Stopping simulation.")
                     simulation_running = False
                     socketio.emit('simulation_stopped', {'message': 'Error setting up next round.'})
                     break # Exit loop
            else:
                 # Round continues, emit state periodically
                 loop_count += 1
                 current_time = time.monotonic()
                 if loop_count % emit_interval_steps == 0 or (current_time - last_emit_time) > min_emit_interval_time:
                      emit_state()
                      last_emit_time = current_time

            # Calculate delay for desired simulation speed
            loop_end_time = time.monotonic()
            elapsed_time = loop_end_time - loop_start_time
            target_delay_seconds = current_config['SIMULATION_SPEED_MS'] / 1000.0
            # Ensure a small positive delay to yield control
            delay = max(0.001, target_delay_seconds - elapsed_time)
            # Use eventlet.sleep for cooperative yielding
            eventlet.sleep(delay) # Use eventlet.sleep directly

        except Exception as e:
            print(f"Error occurred in simulation loop: {e}")
            traceback.print_exc()
            simulation_running = False
            socketio.emit('simulation_stopped', {'message': f'Runtime Error: {e}'})
            break # Exit loop on error

    print("Simulation loop finished.")
    emit_state() # Emit final state
    simulation_loop_task = None # Clear task variable


def emit_state():
    """ Safely gets and emits the current game state """
    try:
        state = get_game_state()
        socketio.emit('update_state', state)
    except Exception as e:
        print(f"Error emitting state: {e}")
        traceback.print_exc()


# ================================================================
# --- Flask Routes & SocketIO Events ---
# ================================================================
@app.route('/')
def index():
    """ Serves the main HTML page """
    try:
        # Use render_template to serve the HTML file from the template folder
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return "Error loading simulation page. Check server logs.", 500


@socketio.on('connect')
def handle_connect():
    """ Handles new client connections """
    sid = request.sid
    print(f"Client connected: {sid}")
    try:
        # Ensure simulation is setup if this is the first connection
        if environment is None or not bots:
             print("First connection, ensuring initial setup...")
             if not setup_simulation(full_reset=True):
                  print(f"Initial setup failed for connecting client {sid}.")
                  emit('status_update', {'message': 'Error: Server simulation setup failed.'}, room=sid)
                  return # Don't send state if setup fails

        # Send the current state (which now includes the full config)
        state = get_game_state()
        emit('initial_state', state, room=sid)
        print(f"Initial state (with full config) sent to {sid}")
        # Client will handle rejoin attempt after receiving initial state
    except Exception as e:
        print(f"Error sending initial data to {sid}: {e}")
        traceback.print_exc()


@socketio.on('disconnect')
def handle_disconnect():
    """ Handles client disconnections, releasing player control """
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in players:
        player_data = players.pop(sid)
        original_bot_id = player_data['original_bot_id']
        print(f"Player {sid} released control of bot {original_bot_id}")
        if original_bot_id in bots:
            # Revert bot back to AI control
            bots[original_bot_id]['is_player_controlled'] = False
            bots[original_bot_id]['target_coordinate'] = None # Clear target
            bots[original_bot_id]['mode'] = "AI Control" # Update mode
            # Inform other clients that the bot is no longer player controlled
            socketio.emit('player_left', {'player_id': original_bot_id})
            emit_state() # Update everyone's state
    # Clear any pending direct actions from this player
    if sid in player_direct_actions:
        del player_direct_actions[sid]


@socketio.on('join_game')
def handle_join_game(data=None): # Added data=None for robustness
    """ Allows a client to take control of a specific available Learning Bot """
    sid = request.sid
    if sid in players:
        emit('join_ack', {'success': False, 'message': 'You are already controlling a bot.'}, room=sid)
        return

    if not environment or not bots:
        emit('join_ack', {'success': False, 'message': 'Simulation not initialized.'}, room=sid)
        return

    target_bot_id = data.get('target_bot_id') if data else None

    if not target_bot_id:
        emit('join_ack', {'success': False, 'message': 'No target bot specified.'}, room=sid)
        return

    # Check if the target bot exists, is Learning, and is not already controlled
    if target_bot_id in bots and bots[target_bot_id]['type'] == 'Learning' and not bots[target_bot_id].get('is_player_controlled', False):
        print(f"Player {sid} taking control of Learning Bot {target_bot_id}")
        # Mark the bot as player controlled
        bots[target_bot_id]['is_player_controlled'] = True
        bots[target_bot_id]['target_coordinate'] = None # Clear any previous target
        bots[target_bot_id]['mode'] = "Player Control"
        # Store the association
        players[sid] = {'player_bot_id': target_bot_id, 'original_bot_id': target_bot_id}
        # Acknowledge success to the player
        emit('join_ack', {'success': True, 'player_id': target_bot_id}, room=sid)
        # Inform other clients (and update state for everyone)
        socketio.emit('player_joined', {'player_id': target_bot_id})
        emit_state()
    else:
        # Bot not available or doesn't exist
        message = f"Bot {target_bot_id} is not available for control."
        if target_bot_id not in bots: message = f"Bot {target_bot_id} does not exist."
        elif bots[target_bot_id]['type'] != 'Learning': message = f"Bot {target_bot_id} is not a Learning bot."
        elif bots[target_bot_id].get('is_player_controlled', False): message = f"Bot {target_bot_id} is already controlled."

        emit('join_ack', {'success': False, 'message': message}, room=sid)
        print(f"Player {sid} failed to join {target_bot_id}: {message}")


@socketio.on('rejoin_game')
def handle_rejoin_game(data):
    """ Allows a client to attempt to regain control after reload """
    sid = request.sid
    if sid in players:
        emit('rejoin_ack', {'success': False, 'message': 'Already controlling a bot.'}, room=sid)
        return

    target_bot_id = data.get('playerBotId')
    if not target_bot_id:
        emit('rejoin_ack', {'success': False, 'message': 'No target bot ID provided.'}, room=sid)
        return

    print(f"Player {sid} attempting to rejoin control of bot {target_bot_id}")

    if target_bot_id in bots and bots[target_bot_id]['type'] == 'Learning':
        # Check if *another* player is controlling it
        already_controlled_by_other = False
        for other_sid, p_data in players.items():
            if other_sid != sid and p_data['player_bot_id'] == target_bot_id:
                already_controlled_by_other = True
                break

        if not already_controlled_by_other:
            # Bot exists, is Learning, and is not currently controlled by someone else - grant control
            bots[target_bot_id]['is_player_controlled'] = True
            bots[target_bot_id]['target_coordinate'] = None # Clear target on rejoin
            bots[target_bot_id]['mode'] = "Player Control (Rejoin)"
            players[sid] = {'player_bot_id': target_bot_id, 'original_bot_id': target_bot_id}
            emit('rejoin_ack', {'success': True, 'player_id': target_bot_id}, room=sid)
            socketio.emit('player_joined', {'player_id': target_bot_id})
            emit_state()
            print(f"Player {sid} successfully rejoined control of {target_bot_id}")
        else:
            # Bot is already controlled by someone else
            emit('rejoin_ack', {'success': False, 'message': f'Bot {target_bot_id} is already controlled by another player.'}, room=sid)
            print(f"Player {sid} failed to rejoin: Bot {target_bot_id} already controlled.")
    else:
        # Bot doesn't exist or isn't a Learning bot
        emit('rejoin_ack', {'success': False, 'message': f'Bot {target_bot_id} not available for control.'}, room=sid)
        print(f"Player {sid} failed to rejoin: Bot {target_bot_id} not available.")


@socketio.on('leave_game')
def handle_leave_game(data=None):
    """ Allows a player to relinquish control of their bot """
    sid = request.sid
    if sid in players:
        player_data = players.pop(sid)
        original_bot_id = player_data['original_bot_id']
        print(f"Player {sid} leaving control of bot {original_bot_id}")
        if original_bot_id in bots:
            bots[original_bot_id]['is_player_controlled'] = False
            bots[original_bot_id]['target_coordinate'] = None # Clear target
            bots[original_bot_id]['mode'] = "AI Control"
            socketio.emit('player_left', {'player_id': original_bot_id})
            emit_state()
        emit('leave_ack', {'success': True}, room=sid) # Acknowledge leave
        # Clear any pending direct actions from this player
        if sid in player_direct_actions:
            del player_direct_actions[sid]
    else:
        emit('leave_ack', {'success': False, 'message': 'Not currently controlling a bot.'}, room=sid)


@socketio.on('player_action')
def handle_player_action(data):
    """ Receives a DIRECT action from a player client (e.g., mobile buttons) """
    sid = request.sid
    if sid in players:
        player_data = players[sid]
        player_bot_id = player_data['player_bot_id']
        if player_bot_id in bots and bots[player_bot_id].get('is_player_controlled', False):
            action = data.get('action')
            try:
                action_int = int(action)
                # Validate action range
                if 0 <= action_int < current_config['NUM_ACTIONS']:
                    # Store the direct action to be processed in the next simulation step
                    player_direct_actions[sid] = action_int
                    # print(f"Received direct action {action_int} for player {sid} (bot {player_bot_id})") # Debug
                else:
                    print(f"Warning: Invalid action value {action} received from player {sid} for bot {player_bot_id}")
            except (ValueError, TypeError):
                print(f"Warning: Non-integer action '{action}' received from player {sid} for bot {player_bot_id}")
        # else: Player might have disconnected just before action arrived or lost control
    # else: Action received from non-player or disconnected player

@socketio.on('update_player_target')
def handle_update_player_target(data):
    """ Receives a target coordinate from a player client (e.g., canvas click/drag) """
    sid = request.sid
    if sid in players:
        player_data = players[sid]
        player_bot_id = player_data['player_bot_id']
        if player_bot_id in bots and bots[player_bot_id].get('is_player_controlled', False):
            target = data.get('target') # Expects {x: num, y: num} or null
            if target is None:
                bots[player_bot_id]['target_coordinate'] = None
                # print(f"Player {sid} cleared target for bot {player_bot_id}") # Debug
            elif isinstance(target, dict) and 'x' in target and 'y' in target:
                try:
                    tx, ty = int(target['x']), int(target['y'])
                    # Validate coordinates against grid size
                    if 0 <= tx < current_config['GRID_SIZE'] and 0 <= ty < current_config['GRID_SIZE']:
                        bots[player_bot_id]['target_coordinate'] = {'x': tx, 'y': ty}
                        # print(f"Player {sid} set target for bot {player_bot_id} to ({tx}, {ty})") # Debug
                    else:
                        print(f"Warning: Invalid target coordinates ({tx}, {ty}) received from player {sid}")
                        bots[player_bot_id]['target_coordinate'] = None # Clear if invalid
                except (ValueError, TypeError):
                    print(f"Warning: Non-integer target coordinates received from player {sid}: {target}")
                    bots[player_bot_id]['target_coordinate'] = None # Clear if invalid type
            else:
                 print(f"Warning: Invalid target format received from player {sid}: {target}")
                 bots[player_bot_id]['target_coordinate'] = None # Clear if invalid format


@socketio.on('start_simulation')
def handle_start_simulation(data=None): # Added data=None
    """ Starts the simulation loop """
    global simulation_running, simulation_loop_task
    if simulation_running:
        print("Warning: Start simulation request received, but already running.")
        emit('status_update', {'message': 'Simulation is already running.'}, room=request.sid)
        return
    print("Start simulation request received.")
    # Ensure setup is complete before starting
    if environment is None or not bots:
         print("Environment/bots not ready, attempting setup before starting...")
         if not setup_simulation(full_reset=False): # Try a round reset first
              socketio.emit('simulation_stopped', {'message': 'Initialization failed on start attempt.'})
              print("Error: Setup failed before starting.")
              return

    simulation_running = True
    # Start the background task if it's not already running or has finished
    if simulation_loop_task is None or (hasattr(simulation_loop_task, 'dead') and simulation_loop_task.dead):
         print("Starting background simulation loop task.")
         # Use the corrected function signature (no async def)
         simulation_loop_task = socketio.start_background_task(simulation_loop)
    else:
         print("Warning: Simulation task object exists and might be active. Not restarting.")

    socketio.emit('simulation_started')
    emit_state() # Emit the state immediately after starting


@socketio.on('stop_simulation')
def handle_stop_simulation(data=None): # Added data=None
    """ Stops the simulation loop """
    global simulation_running
    if not simulation_running:
        print("Warning: Stop simulation request received, but not running.")
        emit('status_update', {'message': 'Simulation is already stopped.'}, room=request.sid)
        return
    print("Stop simulation request received.")
    simulation_running = False # Signal the loop to stop
    # The loop itself will emit the final state and clear the task variable
    # Emit stopped status immediately for responsiveness
    socketio.emit('simulation_stopped', {'message': 'Stopped.'})


@socketio.on('reset_round')
def handle_reset_round(data=None): # Added data=None
    """ Resets the current round """
    global simulation_running
    was_running = simulation_running
    print("Reset round request received.")
    if simulation_running:
        handle_stop_simulation()
        eventlet.sleep(0.1) # Allow time for loop to stop

    if setup_simulation(full_reset=False):
        emit_state()
        status_msg = 'New Round Ready.' + (' Press Start to run.' if was_running else '')
        socketio.emit('status_update', {'message': status_msg})
        print("New round setup complete.")
    else:
        socketio.emit('status_update', {'message': 'Error resetting round.'})
        print("Error during round reset.")


@socketio.on('reset_full')
def handle_reset_full(data=None): # Added data=None
    """ Performs a full reset of the simulation and learning state """
    global simulation_running
    was_running = simulation_running
    print("Full reset request received.")
    if simulation_running:
        handle_stop_simulation()
        eventlet.sleep(0.1) # Allow time for loop to stop

    if setup_simulation(full_reset=True):
        emit_state()
        status_msg = 'Full Reset Complete. Ready.' + (' Press Start to run.' if was_running else '')
        socketio.emit('status_update', {'message': status_msg})
        print("Full reset setup complete.")
    else:
        socketio.emit('status_update', {'message': 'Error during full reset.'})
        print("Error during full reset.")


@socketio.on('update_config')
def handle_update_config(data):
    """ Updates the simulation configuration based on client request """
    global current_config
    if simulation_running:
        emit('config_update_ack', {'success': False, 'message': 'Cannot update config while simulation is running.'}, room=request.sid)
        return

    try:
        new_config_data = data.get('config', {})
        needs_full_reset = False
        needs_round_reset = False
        changed_keys = []
        print("Received config update request:", new_config_data)

        # Create a temporary copy to validate changes
        temp_config = copy.deepcopy(current_config)

        # Define which keys trigger which type of reset
        reset_all_keys = ['GRID_SIZE', 'NUM_HC_BOTS', 'NUM_LEARNING_BOTS', 'NUM_GOALS', 'LEARNING_BOT_DIM', 'LEARNING_BOT_MEM_DEPTH', 'LEARNING_BOT_LR', 'LEARNING_BOT_WEIGHT_DECAY', 'LEARNING_BOT_MOMENTUM', 'LEARNING_BOT_MAX_GRAD_NORM']
        reset_round_keys = ['MAX_STEPS_PER_ROUND', 'VISIBILITY_RANGE', 'OBSTACLES_FACTOR_MIN', 'OBSTACLES_FACTOR_MAX', 'MIN_GOAL_START_DISTANCE_FACTOR', 'MIN_BOT_START_DISTANCE_FACTOR', 'MIN_BOT_GOAL_DISTANCE_FACTOR']
        # Keys that can be updated immediately (affect next step)
        immediate_update_keys = ['SIMULATION_SPEED_MS', 'FREEZE_DURATION', 'LEARNING_BOT_BASE_EXPLORATION_RATE', 'LEARNING_BOT_RULE_EXPLORE_PERCENT']

        for key, value in new_config_data.items():
            if key in DEFAULT_CONFIG:
                try:
                    # Convert value to the correct type based on default config
                    default_type = type(DEFAULT_CONFIG[key])
                    if value is None: continue # Skip null values

                    if default_type is int: converted_value = int(round(float(value)))
                    elif default_type is float: converted_value = float(value)
                    else: converted_value = default_type(value) # e.g., str

                    # Apply constraints (match constraints defined in JS)
                    if key == "GRID_SIZE": converted_value = max(10, min(120, converted_value))
                    if key == "NUM_GOALS": converted_value = max(0, min(150, converted_value))
                    if key == "NUM_HC_BOTS": converted_value = max(0, min(50, converted_value))
                    if key == "NUM_LEARNING_BOTS": converted_value = max(0, min(50, converted_value)) # Ensure at least 0
                    if key == "MAX_STEPS_PER_ROUND": converted_value = max(100, min(10000, converted_value))
                    if key == "VISIBILITY_RANGE": converted_value = max(2, min(50, converted_value))
                    if key == "LEARNING_BOT_DIM":
                        # Ensure DIM is a power of 2 or multiple of common step (e.g., 16) for potential hardware reasons
                        # Make it at least 32, max 1024
                        converted_value = max(32, min(1024, (converted_value // 16) * 16 if converted_value > 32 else 32))
                    if key == "LEARNING_BOT_MEM_DEPTH": converted_value = max(1, min(8, converted_value))
                    if key == "LEARNING_BOT_LR": converted_value = max(1e-5, min(0.1, converted_value))
                    if key == "LEARNING_BOT_WEIGHT_DECAY": converted_value = max(0.0, min(0.1, converted_value))
                    if key == "LEARNING_BOT_MOMENTUM": converted_value = max(0.5, min(0.999, converted_value))
                    if key == "LEARNING_BOT_MAX_GRAD_NORM": converted_value = max(0.1, min(10.0, converted_value))
                    if key == "LEARNING_BOT_BASE_EXPLORATION_RATE": converted_value = max(0.0, min(1.0, converted_value))
                    if key == "LEARNING_BOT_RULE_EXPLORE_PERCENT": converted_value = max(0.0, min(1.0, converted_value))
                    if key == "SIMULATION_SPEED_MS": converted_value = max(1, min(1000, converted_value))
                    if key == "FREEZE_DURATION": converted_value = max(1, min(100, converted_value))
                    # Add constraints for obstacle factors etc. if needed
                    if key == "OBSTACLES_FACTOR_MIN": converted_value = max(0.0, min(0.5, converted_value))
                    if key == "OBSTACLES_FACTOR_MAX": converted_value = max(float(temp_config.get('OBSTACLES_FACTOR_MIN',0.0)), min(0.7, converted_value))


                    # Check if the value actually changed
                    current_value = temp_config.get(key)
                    is_different = False
                    if isinstance(converted_value, float) and isinstance(current_value, float):
                         is_different = abs(converted_value - current_value) > 1e-9 # Tolerance for float comparison
                    else:
                         is_different = current_value != converted_value

                    if is_different:
                        print(f"Applying config change: {key}: {current_value} -> {converted_value}")
                        temp_config[key] = converted_value
                        changed_keys.append(key)
                        # Determine reset level needed
                        if key in reset_all_keys: needs_full_reset = True
                        elif key in reset_round_keys: needs_round_reset = True

                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid type or value for parameter '{key}'. Value '{value}'. Skipping. Error: {e}")
                    continue # Skip this parameter

        if changed_keys:
             # Apply the validated changes to the actual config
             current_config = temp_config
             print(f"Configuration updated. Changed keys: {changed_keys}")
             # Determine final reset requirement
             if needs_full_reset: needs_round_reset = True # Full reset implies round reset
             print(f"Reset required: Full={needs_full_reset}, Round={needs_round_reset}")

             # If NN parameters changed, update the manager *now* before acknowledging
             nn_params_changed = any(k in ['LEARNING_BOT_DIM', 'LEARNING_BOT_MEM_DEPTH', 'LEARNING_BOT_LR', 'LEARNING_BOT_WEIGHT_DECAY', 'LEARNING_BOT_MOMENTUM', 'LEARNING_BOT_MAX_GRAD_NORM'] for k in changed_keys)
             if nn_params_changed:
                 print("NN parameters changed, updating manager...")
                 update_neural_memory_manager()
                 if not neural_memory_manager:
                      # Critical error if NN manager fails
                      emit('config_update_ack', {'success': False, 'message': 'Error: Failed to update NN manager with new parameters.'}, room=request.sid)
                      return # Don't proceed

             # Acknowledge success and required resets
             emit('config_update_ack', {
                 'success': True,
                 'needs_full_reset': needs_full_reset,
                 'needs_round_reset': needs_round_reset,
                 'updated_config': current_config # Send back the final applied config
             }, room=request.sid)
             # Broadcast the new config to all clients
             socketio.emit('config_update', current_config)
        else:
             print("No effective configuration changes detected.")
             emit('config_update_ack', {
                 'success': True, 'needs_full_reset': False, 'needs_round_reset': False,
                 'updated_config': current_config
             }, room=request.sid)

    except Exception as e:
        print(f"Error updating configuration: {e}")
        traceback.print_exc()
        emit('config_update_ack', {'success': False, 'message': f'Internal server error during config update: {e}'}, room=request.sid)


# ================================================================
# --- Server Start ---
# ================================================================
if __name__ == '__main__':
    print("Initializing simulation state...")
    update_neural_memory_manager() # Initialize NN manager at start
    if not neural_memory_manager:
        print("CRITICAL: Neural Memory Manager failed to initialize. Check PyTorch/CUDA setup. Exiting.")
        exit(1)

    if not setup_simulation(full_reset=True):
        print("CRITICAL: Initial simulation setup failed. Check environment parameters. Exiting.")
        exit(1)
    else:
        print("Initial setup successful.")

    port = int(os.environ.get('PORT', 5001))
    print(f"Attempting to start server on http://0.0.0.0:{port}")
    try:
        print("Starting Flask-SocketIO server with eventlet...")
        # Use use_reloader=False to prevent issues with multiprocessing/CUDA in reloader
        socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False)
        print("Server stopped.")
    except OSError as e:
         if "Address already in use" in str(e):
             print(f"Error: Port {port} is already in use. Please free the port or choose a different one.")
         else:
             print(f"Error: Failed to start server due to OS error: {e}")
             traceback.print_exc()
    except Exception as e:
        print(f"Error: An unexpected error occurred during server startup: {e}")
        traceback.print_exc()

