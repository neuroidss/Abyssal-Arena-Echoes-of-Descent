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

# --- PyTorch Imports (Only if needed directly, usually library handles it) ---
# import torch # No direct torch use here if library is fully self-contained

# --- FastAPI Imports ---
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # If serving static files

# --- Local Library Import ---
try:
    from learning_library import CorticalLearningLibrary
except ImportError:
    print("\n" + "="*60)
    print("ERROR: Failed to import 'learning_library.py'.")
    print("Ensure 'learning_library.py' is in the same directory as 'server.py'.")
    print("Simulation cannot run without the learning library.")
    print("="*60 + "\n")
    CorticalLearningLibrary = None # Set to None to prevent runtime errors later

# --- Configuration (Defaults & Global State) ---

DEFAULT_CONFIG = {
    # Simulation Params
    "GRID_SIZE": 35, "NUM_HC_BOTS": 1, "NUM_LEARNING_BOTS": 3, "NUM_GOALS": 15,
    "MAX_OBSTACLES_FACTOR": 0.08, "MIN_OBSTACLES_FACTOR": 0.03, "MAX_STEPS_PER_ROUND": 2500,
    "SIMULATION_SPEED_MS": 20, # Can be faster with GPU
    "FREEZE_DURATION": 25,
    "VISIBILITY_RANGE": 8,
    "LEARNING_BOT_BASE_EXPLORATION_RATE": 0.10, # Start with less exploration
    "LEARNING_BOT_RULE_EXPLORE_PERCENT": 0.60,
    "NUM_ACTIONS": 6, # 0:Up, 1:Left, 2:Right, 3:Down, 4:Punch, 5:ClaimGoal

    # --- Feature Encoding Details (MUST BE CONSISTENT) ---
    # These define the structure of the vector fed TO the library
    "FEATURE_SCALAR_NAMES": ['wallDistance', 'nearestVisibleGoalDist', 'numVisibleGoals', 'nearestOpponentDist'],
    "FEATURE_BINARY_NAMES": ['isFrozen', 'opponentIsFrozen'],
    "FEATURE_CATEGORY_LAST_ACTION": True, # Include last action index
    "FEATURE_CATEGORY_OPPONENT_TYPE": True, # Include opponent type index
    # Action indices range from -1 (invalid) to NUM_ACTIONS-1. We need NUM_ACTIONS+1 categories.
    # Opponent types: Hardcoded, Learning, None. 3 categories.
    "FEATURE_LAST_ACTION_DIM": 7, # NUM_ACTIONS + 1 (for -1 invalid state)
    "FEATURE_OPPONENT_TYPE_DIM": 3, # Hardcoded, Learning, None

    # --- NEW: Cortical Learning Library Params (Require Reset All) ---
    "LIB_DEVICE": "auto", # "auto", "cuda", "cpu"
    # LIB_INPUT_VECTOR_SIZE is calculated below based on features
    "LIB_RNN_TYPE": "LSTM", # "LSTM" or "GRU"
    "LIB_HIDDEN_SIZE": 512, # Increased default
    "LIB_NUM_LAYERS": 3,    # Increased default
    "LIB_DROPOUT": 0.1,
    "LIB_LEARNING_RATE": 0.0005, # Tuned default
    "LIB_LOSS_TYPE": "MSE", # "MSE" or "L1" or "Huber"

    # Env Generation Params
    "MIN_GOAL_START_DISTANCE_FACTOR": 0.25, "MIN_BOT_START_DISTANCE_FACTOR": 0.35,
    "MIN_BOT_GOAL_DISTANCE_FACTOR": 0.20,
}

# --- Calculate Library Input Size ---
def calculate_input_vector_size(config):
    size = 0
    size += len(config.get("FEATURE_SCALAR_NAMES", []))
    size += len(config.get("FEATURE_BINARY_NAMES", []))
    if config.get("FEATURE_CATEGORY_LAST_ACTION", False):
        # One-hot encoding for action
        size += config.get("FEATURE_LAST_ACTION_DIM", config.get("NUM_ACTIONS", 6) + 1)
    if config.get("FEATURE_CATEGORY_OPPONENT_TYPE", False):
        # One-hot encoding for opponent type
        size += config.get("FEATURE_OPPONENT_TYPE_DIM", 3)
    return size

DEFAULT_CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(DEFAULT_CONFIG)
# --- End Input Size Calculation ---


CONFIG = DEFAULT_CONFIG.copy()
CONFIG_FILE = "simulation_config_v6.json" # New version suffix

# --- Global Simulation State ---
environment: Optional['GridEnvironment'] = None
hardcoded_bots: List['HardcodedBot'] = []
learning_bots: List['LearningBot'] = []
all_bots: List[Any] = []
cortical_library: Optional['CorticalLearningLibrary'] = None # THE single shared library instance

simulation_task: Optional[asyncio.Task] = None
is_running: bool = False
round_number: int = 0
hc_total_goals: int = 0
learning_total_goals: int = 0
_simulation_lock = asyncio.Lock() # Protects simulation state modifications

# --- WebSocket Connection Management ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
             if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_text(message)
        except Exception as e:
             print(f"Error sending personal message: {e}")
             self.disconnect(websocket) # Assume disconnect on error

    async def broadcast(self, message: str):
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)
                elif connection.client_state != WebSocketState.CONNECTING:
                     disconnected_clients.append(connection)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                disconnected_clients.append(connection) # Assume disconnect on error
        for client in disconnected_clients:
             self.disconnect(client)

manager = ConnectionManager()

# --- Utility Functions ---
def manhattan_distance(pos1: Dict[str, int], pos2: Dict[str, int]) -> int:
    if not pos1 or not pos2: return float('inf')
    return abs(pos1.get('x', 0) - pos2.get('x', 0)) + abs(pos1.get('y', 0) - pos2.get('y', 0))

def get_opponent_type_index(opponent_type_name: Optional[str]) -> int:
    """Maps opponent type name to a fixed index for one-hot encoding."""
    if opponent_type_name == 'Hardcoded': return 0
    elif opponent_type_name == 'Learning': return 1
    else: return 2 # 'None' or unknown

# --- Parameter Loading/Saving (Check library compatibility) ---
def load_params_from_file():
    global CONFIG
    if not os.path.exists(CONFIG_FILE):
        print(f"Config file {CONFIG_FILE} not found. Using defaults.")
        CONFIG = DEFAULT_CONFIG.copy()
        return False, False # Params not loaded, lib compatibility unknown (use fresh)

    try:
        with open(CONFIG_FILE, 'r') as f:
            loaded_params = json.load(f)

        validated_config = DEFAULT_CONFIG.copy()
        # Check if essential structural params for library exist in loaded file
        library_compatible = True
        critical_lib_keys = ['LIB_RNN_TYPE', 'LIB_HIDDEN_SIZE', 'LIB_NUM_LAYERS', 'LIB_INPUT_VECTOR_SIZE']
        for key in critical_lib_keys:
             if key not in loaded_params or loaded_params[key] != DEFAULT_CONFIG[key]:
                 print(f"Warning: Critical library parameter '{key}' mismatch or missing in saved config.")
                 print(f"  Saved: {loaded_params.get(key)}, Default: {DEFAULT_CONFIG[key]}")
                 library_compatible = False
                 # Keep default for this key if mismatch
                 #validated_config[key] = DEFAULT_CONFIG[key] #This happens anyway

        # Load other params, validating types
        for key, default_value in DEFAULT_CONFIG.items():
             if key in loaded_params and type(loaded_params[key]) == type(default_value):
                  validated_config[key] = loaded_params[key]
             elif key in loaded_params:
                  print(f"Warning: Loaded param '{key}' type mismatch ({type(loaded_params[key])} vs {type(default_value)}) or deprecated. Using default.")
                  # validated_config[key] = default_value # Already set

        CONFIG = validated_config
        # Recalculate input size just in case feature flags changed (though they shouldn't typically)
        CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)

        print("Parameters loaded from", CONFIG_FILE)
        if not library_compatible:
             print("WARNING: Loaded parameters are incompatible with the current library structure definition.")
             print("         Saved library state (if any) will NOT be loaded.")
        return True, library_compatible # Params loaded, compatibility flag

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading parameters from {CONFIG_FILE}: {e}. Using defaults.")
        CONFIG = DEFAULT_CONFIG.copy()
        CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
        return False, False
    except Exception as e:
        print(f"Unexpected error loading parameters: {e}. Using defaults."); traceback.print_exc()
        CONFIG = DEFAULT_CONFIG.copy()
        CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
        return False, False


def save_params_to_file():
    global CONFIG
    # Ensure calculated size is correct before saving
    CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        print("Parameters saved to", CONFIG_FILE); return True
    except Exception as e:
        print(f"Error saving parameters: {e}"); return False

def reset_params_to_default():
    global CONFIG
    CONFIG = DEFAULT_CONFIG.copy()
    # Recalculate size after reset
    CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
    print("Parameters reset to default.")

# --- Encode Sensory Data into Agnostic Vector ---
def encode_senses_to_vector(senses: Dict, last_action: int, config: Dict) -> np.ndarray:
    """Encodes sensory dictionary and last action into a flat numpy vector."""
    vector = []
    vis_range = config['VISIBILITY_RANGE']
    num_actions = config['NUM_ACTIONS']

    # 1. Scalars (Normalized)
    for name in config['FEATURE_SCALAR_NAMES']:
        value = senses.get(name, 0.0)
        if name.endswith('Dist') or name.endswith('Distance'):
            # Treat inf/large values as max range, normalize 0 to 1
            norm_val = min(float(value), vis_range + 1) / max(1.0, vis_range + 1) if isinstance(value, (int, float)) and value != float('inf') else 1.0
        elif name == 'numVisibleGoals':
            norm_val = min(float(value), 10.0) / 10.0 # Assume max ~10 visible is reasonable upper bound
        else:
            norm_val = float(value) # Fallback
        vector.append(norm_val)

    # 2. Binaries (0.0 or 1.0)
    for name in config['FEATURE_BINARY_NAMES']:
        vector.append(1.0 if senses.get(name, False) else 0.0)

    # 3. Last Action (One-Hot)
    if config['FEATURE_CATEGORY_LAST_ACTION']:
        action_dim = config['FEATURE_LAST_ACTION_DIM']
        action_one_hot = np.zeros(action_dim)
        # Map action (-1 to num_actions-1) to index (0 to num_actions)
        action_index = last_action + 1
        if 0 <= action_index < action_dim:
            action_one_hot[action_index] = 1.0
        else: # Should not happen if action is valid, but handle potential errors
             action_one_hot[0] = 1.0 # Use index 0 for invalid/out-of-bounds
        vector.extend(action_one_hot.tolist())

    # 4. Opponent Type (One-Hot)
    if config['FEATURE_CATEGORY_OPPONENT_TYPE']:
        opponent_type_name = senses.get('opponentType', 'None')
        type_index = get_opponent_type_index(opponent_type_name)
        type_dim = config['FEATURE_OPPONENT_TYPE_DIM']
        type_one_hot = np.zeros(type_dim)
        if 0 <= type_index < type_dim:
            type_one_hot[type_index] = 1.0
        else: # Should not happen
             type_one_hot[2] = 1.0 # Index 2 is 'None'
        vector.extend(type_one_hot.tolist())

    encoded_vector = np.array(vector, dtype=np.float32)

    # --- Final Size Check ---
    expected_size = config['LIB_INPUT_VECTOR_SIZE']
    if encoded_vector.size != expected_size:
        print(f"FATAL ENCODING ERROR: Encoded size {encoded_vector.size} != Expected {expected_size}")
        print("  Senses:", senses)
        print("  Last Action:", last_action)
        print("  Config:", {k:v for k,v in config.items() if k.startswith("FEATURE")})
        # Raise error or return None to halt processing? Returning None is safer.
        raise ValueError(f"Encoding size mismatch: {encoded_vector.size} vs {expected_size}")
        # return None # Or raise error

    return encoded_vector

# --- Decode Action from Predicted Vector ---
def decode_action_from_prediction(prediction_vector: np.ndarray, config: Dict) -> int:
    """Decodes the predicted action index from the library's output vector."""
    if prediction_vector is None: return -1 # Handle None input

    # Find the start index of the action one-hot block
    action_start_index = 0
    action_start_index += len(config['FEATURE_SCALAR_NAMES'])
    action_start_index += len(config['FEATURE_BINARY_NAMES'])

    if not config['FEATURE_CATEGORY_LAST_ACTION']:
        # If action wasn't encoded, cannot decode it. Return random valid move.
        print("Warning: Trying to decode action, but it wasn't encoded. Returning random move.")
        return random.randrange(4)

    action_dim = config['FEATURE_LAST_ACTION_DIM']
    action_end_index = action_start_index + action_dim

    if action_end_index > len(prediction_vector):
         print(f"Error: Action decode index out of bounds. End index {action_end_index}, vector len {len(prediction_vector)}")
         return random.randrange(4) # Fallback

    action_one_hot_part = prediction_vector[action_start_index:action_end_index]

    # Find the index with the highest value in the predicted one-hot block
    predicted_action_index = np.argmax(action_one_hot_part)

    # Map index (0 to num_actions) back to action (-1 to num_actions-1)
    predicted_action = predicted_action_index - 1

    # Validate range
    if not (-1 <= predicted_action < config['NUM_ACTIONS']):
        print(f"Warning: Decoded action {predicted_action} out of valid range [-1, {config['NUM_ACTIONS']-1}]. Falling back.")
        # Fallback: choose a random *movement* action
        return random.randrange(4)

    return predicted_action


# ================================================================
# --- Simulation Environment (Mostly Unchanged) ---
# ================================================================
class GridEnvironment:
    # --- Identical to previous version (server.py v5) ---
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
        total_bots = self.num_hc_bots + self.num_learning_bots
        total_cells = self.size * self.size
        requested_items = total_bots + self.num_goals # Total non-obstacle items requested

        print(f"Randomizing Env (Unconstrained): Size={self.size}x{self.size}, Requested Goals={self.num_goals}, HC={self.num_hc_bots}, Lrn={self.num_learning_bots}")

        occupied: Set[str] = set() # Tracks cells used by goals, bots, and obstacles
        available_cells: List[Tuple[int, int]] = [(x, y) for x in range(self.size) for y in range(self.size)]
        random.shuffle(available_cells) # Shuffle all possible grid cells

        def get_next_available_position(occupied_set: Set) -> Optional[Dict]:
            while available_cells:
                x, y = available_cells.pop()
                pos_str = f"{x},{y}"
                if pos_str not in occupied_set:
                    occupied_set.add(pos_str)
                    return {'x': x, 'y': y}
            return None # No more cells left

        # Place Goals
        goal_id_counter = 0; placed_goals = 0
        for _ in range(self.num_goals):
            pos = get_next_available_position(occupied)
            if pos is None: print(f"Warning: Grid full. Placed only {placed_goals}/{self.num_goals} goals."); break
            goal = {**pos, 'id': f"G{goal_id_counter}"}; goal_id_counter += 1; self.goals.append(goal); placed_goals += 1

        # Place Bots
        placed_bots = 0
        bots_to_place = [(f"H{i}", "Hardcoded") for i in range(self.num_hc_bots)] + \
                        [(f"L{i}", "Learning") for i in range(self.num_learning_bots)]
        random.shuffle(bots_to_place)

        for bot_id, bot_type in bots_to_place:
             pos = get_next_available_position(occupied)
             if pos is None: print(f"Warning: Grid full. Placed only {placed_bots}/{total_bots} bots."); break
             self.start_positions.append({**pos, 'type': bot_type, 'id': bot_id}); placed_bots += 1
        self.start_positions.sort(key=lambda sp: (sp['type'], int(sp['id'][1:])))

        # Place Obstacles
        num_obstacles_to_place = random.randint(self.min_obstacles, self.max_obstacles);
        placed_obstacles = 0
        for _ in range(num_obstacles_to_place):
            pos = get_next_available_position(occupied)
            if pos is None: break
            self.obstacles.add(self._pos_to_string(pos)); placed_obstacles += 1

        print(f"Placement Complete: Goals={len(self.goals)}, Bots={len(self.start_positions)}, Obstacles={len(self.obstacles)} / Target:{num_obstacles_to_place}")
        if len(self.goals) < self.num_goals or len(self.start_positions) < total_bots:
            print("Placement Warning: Not all requested goals/bots could be placed due to grid capacity.")

    def is_valid(self, pos: Dict) -> bool:
        x, y = pos.get('x'), pos.get('y'); return 0 <= x < self.size and 0 <= y < self.size and self._pos_to_string(pos) not in self.obstacles

    def get_sensory_data(self, acting_bot: Any, all_bots_list: List[Any], visibility_range: int) -> Dict:
        # Identical to previous version (v5)
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
        # Identical to previous version (v5)
        next_pos = bot_pos.copy()
        if action_index == 0: next_pos['y'] -= 1 # Up
        elif action_index == 1: next_pos['x'] -= 1 # Left
        elif action_index == 2: next_pos['x'] += 1 # Right
        elif action_index == 3: next_pos['y'] += 1 # Down
        return next_pos

    def get_adjacent_unclaimed_goal(self, bot_pos: Dict) -> Optional[Dict]:
        # Identical to previous version (v5)
        for goal in self.goals:
             if goal['id'] not in self.claimed_goals and manhattan_distance(bot_pos, goal) == 1: return goal
        return None

    def claim_goal(self, goal_id: str, bot_id: str) -> bool:
        # Identical to previous version (v5)
        if not goal_id or goal_id in self.claimed_goals: return False
        goal = next((g for g in self.goals if g['id'] == goal_id), None)
        if goal: self.claimed_goals.add(goal_id); print(f"Goal {goal_id} at ({goal['x']},{goal['y']}) claimed by {bot_id}."); return True
        return False

    def are_all_goals_claimed(self) -> bool: return len(self.goals) > 0 and len(self.claimed_goals) >= len(self.goals)
    def get_state_for_client(self) -> Dict: return { 'size': self.size, 'goals': self.goals, 'obstacles': list(self.obstacles), 'claimedGoals': list(self.claimed_goals) }

# ================================================================
# --- Bot Implementations (Hardcoded Unchanged, Learning Updated) ---
# ================================================================

class HardcodedBot:
    # --- Identical to previous version (server.py v5) ---
    def __init__(self, start_pos: Dict, bot_id: str):
        self.id = bot_id; self.type = "Hardcoded"; self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0; self.last_move_attempt = -1; self.stuck_counter = 0
    def reset(self, start_pos: Dict):
        self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0; self.last_move_attempt = -1; self.stuck_counter = 0
    def get_action(self, senses: Dict, env: GridEnvironment) -> int:
        if self.freeze_timer > 0: self.stuck_counter = 0; return -1 # Frozen
        adjacent_goal = env.get_adjacent_unclaimed_goal(self.pos);
        if adjacent_goal: self.stuck_counter = 0; self.last_move_attempt = 5; return 5 # Claim adjacent goal
        nearest_opponent = senses.get('_nearestOpponent');
        if nearest_opponent and senses.get('nearestOpponentDist') == 1 and not senses.get('opponentIsFrozen'): self.stuck_counter = 0; self.last_move_attempt = 4; return 4 # Punch adjacent unfrozen opponent
        visible_goals = senses.get('_visibleGoals', []);
        if visible_goals:
            # --- Pathfinding towards nearest goal ---
            nearest_goal = visible_goals[0]; dx = nearest_goal['x'] - self.pos['x']; dy = nearest_goal['y'] - self.pos['y']; preferred_moves = []
            # Determine preferred move order based on larger distance component
            if abs(dx) > abs(dy):
                if dx != 0: preferred_moves.append(2 if dx > 0 else 1) # Right/Left
                if dy != 0: preferred_moves.append(3 if dy > 0 else 0) # Down/Up
            else:
                if dy != 0: preferred_moves.append(3 if dy > 0 else 0) # Down/Up
                if dx != 0: preferred_moves.append(2 if dx > 0 else 1) # Right/Left
            # Try preferred moves first
            for move_action in preferred_moves:
                next_pos = env.perform_move_action(self.pos, move_action)
                if env.is_valid(next_pos): self.stuck_counter = 0; self.last_move_attempt = move_action; return move_action
            # --- Obstacle Avoidance (Simple: try perpendicular moves if stuck) ---
            self.stuck_counter += 1
            sideway_moves = [0, 3] if abs(dx) >= abs(dy) else [1, 2] # If equal dx/dy, prioritize up/down avoidance
            sideway_moves = [m for m in sideway_moves if m not in preferred_moves] # Get moves not already tried
            random.shuffle(sideway_moves)
            for avoid_action in sideway_moves:
                 next_pos = env.perform_move_action(self.pos, avoid_action)
                 if env.is_valid(next_pos): self.last_move_attempt = avoid_action; return avoid_action
        else: self.stuck_counter = 0 # Reset stuck counter if no goals visible
        # --- Random Movement if no goal visible or stuck ---
        valid_moves = [action for action in range(4) if env.is_valid(env.perform_move_action(self.pos, action))]
        reverse_action = (self.last_move_attempt + 2) % 4 if 0 <= self.last_move_attempt <= 3 else -1
        # Avoid immediately reversing if possible and not stuck for too long
        if len(valid_moves) > 1 and reverse_action in valid_moves and self.stuck_counter < 3:
             non_reverse_moves = [m for m in valid_moves if m != reverse_action]
             if non_reverse_moves: valid_moves = non_reverse_moves
        if valid_moves: chosen_move = random.choice(valid_moves); self.last_move_attempt = chosen_move; return chosen_move
        else: self.last_move_attempt = -1; return -1 # Truly stuck
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
        if not library: raise ValueError("LearningBot requires a valid library instance!")
        self.library = library
        self.config = config # Store reference to global config

        self.pos = start_pos.copy()
        self.steps = 0
        self.goals_reached_this_round = 0
        self.freeze_timer = 0

        self.last_chosen_action = -1 # Action taken that LED to the current state
        self.mode = "Init"
        self.last_loss = 0.0
        self.use_rule_based_exploration = random.random() < self.config.get('LEARNING_BOT_RULE_EXPLORE_PERCENT', 0.6)
        self._hc_logic_provider: Optional[HardcodedBot] = None # For rule-based exploration

        # --- State for Prediction/Training ---
        # Stores the PREDICTION made *in the previous step* for the current step's sensory input
        self.last_prediction_vector: Optional[np.ndarray] = None
        # Stores the INPUT VECTOR that *generated* the last_prediction_vector
        # self.input_vector_for_last_prediction: Optional[np.ndarray] = None # Maybe not needed directly

    def reset(self, start_pos: Dict):
        self.pos = start_pos.copy(); self.steps = 0; self.goals_reached_this_round = 0; self.freeze_timer = 0; self.mode = "Init"; self.last_chosen_action = -1; self.last_loss = 0.0; self._hc_logic_provider = None; self.use_rule_based_exploration = random.random() < self.config.get('LEARNING_BOT_RULE_EXPLORE_PERCENT', 0.6); self.last_prediction_vector = None
        # Reset hidden state in the library for this bot
        self.library.reset_hidden_state(self.id)

    def get_action(self, current_senses: Dict, env: GridEnvironment) -> int:
        """
        Decides the next action using the library prediction or exploration.
        Also prepares the input for the library to predict the *next* step.
        """
        chosen_action = -1

        # I. Encode Current State (using action that LED here) into Input Vector
        try:
            current_input_vector = encode_senses_to_vector(current_senses, self.last_chosen_action, self.config)
            if current_input_vector is None: raise ValueError("Encoding failed")
        except Exception as encode_error:
            print(f"{self.id}: Input Encode Error: {encode_error}"); traceback.print_exc()
            self.last_chosen_action = -1; return -1 # Cannot proceed without valid input

        # II. Get Prediction for *Next* Step from Library
        # The library uses the current_input_vector and the bot's hidden state
        predicted_vector_for_next_step, _ = self.library.process_and_predict(self.id, current_input_vector)

        # Store this prediction - it will be used for TRAINING in the *next* call to train_on_last_step
        self.last_prediction_vector = predicted_vector_for_next_step # Can be None if prediction failed

        # III. Decide Action (Only if NOT frozen)
        if self.freeze_timer > 0:
            self.mode = "Frozen"; chosen_action = -1
        else:
            effective_exploration_rate = self.config.get('LEARNING_BOT_BASE_EXPLORATION_RATE', 0.1)

            if random.random() < effective_exploration_rate:
                # --- Exploration Mode ---
                if self.use_rule_based_exploration:
                    self.mode = "Explore (Rule)";
                    try:
                        # Use temporary HC bot instance for logic
                        if not self._hc_logic_provider: self._hc_logic_provider = HardcodedBot(self.pos, "temp_hc")
                        self._hc_logic_provider.pos = self.pos.copy(); self._hc_logic_provider.freeze_timer = 0; self._hc_logic_provider.last_move_attempt = self.last_chosen_action if 0 <= self.last_chosen_action <= 3 else -1; self._hc_logic_provider.stuck_counter = 0
                        chosen_action = self._hc_logic_provider.get_action(current_senses, env)
                        # Ensure valid action range
                        if not (0 <= chosen_action < self.config['NUM_ACTIONS']): chosen_action = random.randrange(4)
                    except Exception as rule_explore_error: print(f"{self.id} Rule Explore Error: {rule_explore_error}"); chosen_action = random.randrange(4)
                else:
                    self.mode = "Explore (Random)"; chosen_action = random.randrange(self.config['NUM_ACTIONS'])
                    # --- Basic Validation of Random Action ---
                    if chosen_action == 4: # Punch
                         can_punch = current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen')
                         if not can_punch: chosen_action = random.randrange(4) # Reroll move if cannot punch
                    elif chosen_action == 5: # Claim
                         can_claim = env.get_adjacent_unclaimed_goal(self.pos) is not None
                         if not can_claim: chosen_action = random.randrange(4) # Reroll move if cannot claim
                    elif 0 <= chosen_action <= 3: # Move
                        can_move = env.is_valid(env.perform_move_action(self.pos, chosen_action))
                        if not can_move:
                            valid_moves = [mv for mv in range(4) if env.is_valid(env.perform_move_action(self.pos, mv))]; chosen_action = random.choice(valid_moves) if valid_moves else -1
            else:
                # --- Exploitation Mode (Use Library Prediction) ---
                if self.last_prediction_vector is not None:
                    try:
                        predicted_action = decode_action_from_prediction(self.last_prediction_vector, self.config)

                        # --- Validate Predicted Action in Current State ---
                        is_predicted_action_valid = False
                        if 0 <= predicted_action <= 3: # Move
                            is_predicted_action_valid = env.is_valid(env.perform_move_action(self.pos, predicted_action))
                        elif predicted_action == 4: # Punch
                            is_predicted_action_valid = current_senses.get('_nearestOpponent') and current_senses.get('nearestOpponentDist') == 1 and not current_senses.get('opponentIsFrozen')
                        elif predicted_action == 5: # Claim
                            is_predicted_action_valid = env.get_adjacent_unclaimed_goal(self.pos) is not None
                        elif predicted_action == -1: # Predicted 'Invalid/Do Nothing'
                             is_predicted_action_valid = True # Doing nothing is always valid if predicted

                        if is_predicted_action_valid:
                            self.mode = f"Exploit (Predict {predicted_action})"; chosen_action = predicted_action
                        else:
                            # Fallback: Prediction invalid in current context, use HC logic
                            self.mode = f"Exploit (Fallback {predicted_action}->Rule)"
                            try:
                                if not self._hc_logic_provider: self._hc_logic_provider = HardcodedBot(self.pos, "temp_hc")
                                self._hc_logic_provider.pos = self.pos.copy(); self._hc_logic_provider.freeze_timer = 0; self._hc_logic_provider.last_move_attempt = self.last_chosen_action if 0 <= self.last_chosen_action <= 3 else -1; self._hc_logic_provider.stuck_counter = 0
                                chosen_action = self._hc_logic_provider.get_action(current_senses, env)
                                if not (0 <= chosen_action < self.config['NUM_ACTIONS']): chosen_action = random.randrange(4)
                            except Exception as fallback_error: print(f"{self.id} Exploit Fallback Error: {fallback_error}"); chosen_action = random.randrange(4)

                    except Exception as eval_error:
                        print(f"{self.id}: Predict Eval/Decode Error: {eval_error}"); traceback.print_exc(); chosen_action = random.randrange(4); self.mode = "Exploit (Error)"
                else:
                    # Prediction failed in library, fallback to exploration
                    self.mode = "Exploit (No Prediction)"; chosen_action = random.randrange(4) # Simple random move

        # Store the chosen action so it can be encoded in the *next* step's input
        self.last_chosen_action = chosen_action
        return chosen_action

    def update(self, next_pos: Dict, chosen_action: int):
        """Update bot's internal state (position, steps, freeze)."""
        self.pos = next_pos.copy()
        self.steps += 1
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
            if self.freeze_timer == 0:
                 # Just unfrozen, maybe reset library state? Context was lost.
                 print(f"{self.id} unfrozen, resetting library hidden state.")
                 self.library.reset_hidden_state(self.id)
                 self.last_prediction_vector = None # Invalidate prediction

    def train_on_last_step(self, actual_next_senses: Dict):
        """
        Uses the prediction made *before* the step and the actual outcome *after*
        the step to train the library.
        """
        if self.last_prediction_vector is None or self.last_chosen_action == -1:
            # Cannot train if no prediction was made or no action was taken (e.g., first step, frozen, error)
            self.last_loss = 0.0
            return 0.0 # Return 0 loss

        # I. Encode the ACTUAL Next State (using the action *just taken*)
        # This represents the target vector the library *should have* predicted.
        try:
            # The 'last_action' for encoding the target is the action we just took
            actual_next_vector = encode_senses_to_vector(actual_next_senses, self.last_chosen_action, self.config)
            if actual_next_vector is None: raise ValueError("Target Encoding failed")
        except Exception as encode_error:
            print(f"{self.id}: Target Encode Error: {encode_error}"); traceback.print_exc()
            self.last_prediction_vector = None # Invalidate prediction if target failed
            self.last_loss = 0.0
            return 0.0

        # II. Perform Training Step in Library
        # Pass the prediction made *before* the step, and the actual outcome *after* the step.
        try:
            loss = self.library.train_on_step(self.last_prediction_vector, actual_next_vector)
            self.last_loss = loss if loss is not None and loss >= 0 else 0.0 # Store valid loss
        except Exception as train_error:
            print(f"{self.id}: Train Step Error: {train_error}"); traceback.print_exc()
            self.last_loss = 0.0

        # Prediction was used, clear it for the next cycle
        # self.last_prediction_vector = None # No, keep it until the next get_action replaces it

        return self.last_loss

    def apply_freeze(self, duration: int):
        print(f"{self.id} Frozen for {duration} steps!")
        self.freeze_timer = max(self.freeze_timer, duration)
        # Invalidate last prediction if frozen, as context is broken
        self.last_prediction_vector = None
        # Reset hidden state when frozen? Handled in update when timer hits 0.
        # self.library.reset_hidden_state(self.id)

    def get_state_for_client(self) -> Dict:
        return {'id': self.id, 'type': self.type, 'pos': self.pos, 'freezeTimer': self.freeze_timer, 'goalsReachedThisRound': self.goals_reached_this_round, 'steps': self.steps, 'mode': self.mode, 'lastLoss': f"{self.last_loss:.5f}"}


# ================================================================
# --- Simulation Control Logic (Updated for Library & Agnosticism) ---
# ================================================================

async def broadcast_message(msg_dict: Dict):
    """Helper to broadcast JSON messages."""
    await manager.broadcast(json.dumps(msg_dict))

async def send_status_update(message: str, is_error: bool = False):
    await broadcast_message({"type": "status_update", "payload": {"message": message, "is_error": is_error}})

async def send_config_update():
    global CONFIG
    # Ensure calculated size is up-to-date before sending
    CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
    await broadcast_message({"type": "config_update", "payload": CONFIG})

async def send_stats_update():
    avg_loss = None
    if cortical_library: avg_loss = cortical_library.get_average_loss()
    await broadcast_message({
        "type": "stats_update", "payload": { "roundNumber": round_number, "hcTotalGoals": hc_total_goals, "learningTotalGoals": learning_total_goals, "learningAvgLoss": f"{avg_loss:.5f}" if avg_loss is not None else "N/A", }
    })

async def send_library_state_update():
     state_payload = cortical_library.get_internal_state_info() if cortical_library and learning_bots else None
     await broadcast_message({"type": "library_state_update", "payload": state_payload})

def check_config_needs_reset(new_config: Dict) -> Tuple[bool, bool]:
    # --- Check Keys Requiring Reset ---
    needs_full_reset = False; needs_round_reset = False
    full_reset_keys = [
        "GRID_SIZE", "NUM_HC_BOTS", "NUM_LEARNING_BOTS", "NUM_GOALS",
        "VISIBILITY_RANGE", "NUM_ACTIONS",
        "MAX_OBSTACLES_FACTOR", "MIN_OBSTACLES_FACTOR",
        "MIN_GOAL_START_DISTANCE_FACTOR", "MIN_BOT_START_DISTANCE_FACTOR", "MIN_BOT_GOAL_DISTANCE_FACTOR",
        # Feature encoding changes require full reset
        "FEATURE_SCALAR_NAMES", "FEATURE_BINARY_NAMES", "FEATURE_CATEGORY_LAST_ACTION",
        "FEATURE_CATEGORY_OPPONENT_TYPE", "FEATURE_LAST_ACTION_DIM", "FEATURE_OPPONENT_TYPE_DIM",
        # Library structure params require full reset
        "LIB_DEVICE", "LIB_RNN_TYPE", "LIB_HIDDEN_SIZE",
        "LIB_NUM_LAYERS", "LIB_DROPOUT", "LIB_LEARNING_RATE", "LIB_LOSS_TYPE",
        # LIB_INPUT_VECTOR_SIZE change implies feature changes -> full reset
        "LIB_INPUT_VECTOR_SIZE"
    ]
    round_reset_keys = ["MAX_STEPS_PER_ROUND"] # Only this one for now

    # Compare current CONFIG with potential new values in new_config
    for key, current_value in CONFIG.items():
        new_value = new_config.get(key)
        # Check if key exists in new_config, has same type, and different value
        if new_value is not None and type(new_value) == type(current_value) and new_value != current_value:
            if key in full_reset_keys:
                needs_full_reset = True; break # Full reset needed, no need to check further
            elif key in round_reset_keys:
                needs_round_reset = True # Mark for round reset, continue checking for full reset

    if needs_full_reset: needs_round_reset = True # Full reset implies round reset
    return needs_full_reset, needs_round_reset


async def setup_simulation(is_full_reset_request: bool = False, loaded_library_compatible: bool = True):
    """Sets up or resets the simulation environment and learning library."""
    global environment, hardcoded_bots, learning_bots, all_bots
    global cortical_library
    global round_number, hc_total_goals, learning_total_goals

    # Ensure library is available
    if CorticalLearningLibrary is None:
         print("CRITICAL: CorticalLearningLibrary not loaded. Cannot setup simulation.")
         await send_status_update("Error: Learning Library missing!", is_error=True)
         return False

    async with _simulation_lock:
        print(f"--- Setting up Simulation (Full Reset Requested: {is_full_reset_request}, Lib Compatible Load: {loaded_library_compatible}) ---")
        await send_status_update("Status: Initializing...")
        stop_simulation_internal(update_ui=False) # Stop any running loop

        # Determine if reset is needed based on request or config state
        _needs_full, _needs_round = check_config_needs_reset(CONFIG) # Check current config state
        perform_full_reset = is_full_reset_request or _needs_full
        perform_round_reset = perform_full_reset or _needs_round or not environment # Reset round if env missing

        # --- Perform Full Reset Actions (Library, Scores) ---
        if perform_full_reset:
            print("Performing Full Reset (Cortical Library, Scores)...")
            if cortical_library:
                try: cortical_library.dispose() # Release old library resources
                except Exception as e: print(f"Error disposing old library: {e}")
            cortical_library = None # Ensure it's cleared

            try:
                print("Creating new Cortical Learning Library...")
                # Ensure calculated size is correct before creating library
                CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
                cortical_library = CorticalLearningLibrary(CONFIG)

                # Try loading saved state ONLY if compatible with loaded params
                library_state_file = "cortical_library_state.pth"
                if loaded_library_compatible and os.path.exists(library_state_file):
                    print("Attempting to load compatible saved library state...")
                    if not cortical_library.load_state(library_state_file):
                         print("Failed to load state or incompatible structure found in file. Using fresh library.")
                         # If load failed, delete potentially corrupt file?
                         # try: os.remove(library_state_file); print("Deleted problematic state file.")
                         # except OSError as e: print(f"Could not delete problematic state file: {e}")
                    else:
                         print("Loaded saved library state successfully.")
                         # IMPORTANT: Ensure CONFIG reflects loaded library's config
                         global CONFIG
                         CONFIG = cortical_library.config.copy()
                         # Recalculate size based on potentially loaded feature flags
                         CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
                else:
                     print("No compatible saved library state found or loading disabled. Starting fresh.")

            except Exception as error:
                print(f"CRITICAL: Failed to create/load Cortical Library: {error}")
                traceback.print_exc()
                await send_status_update(f"Error: Failed AI init! {error}", is_error=True)
                cortical_library = None # Ensure library is None on failure
                return False

            # Reset scores only on full reset
            round_number = 0; hc_total_goals = 0; learning_total_goals = 0
            print("New Library created/loaded. Scores reset.")

        # Safety check: Ensure library exists after potential full reset
        if not cortical_library:
             print("CRITICAL ERROR: Library instance is missing after setup attempt.")
             await send_status_update("Error: AI Library is missing!", is_error=True)
             return False


        # --- Perform Round Reset Actions (Environment Layout, Bot Positions) ---
        if perform_round_reset:
            print(f"Creating/Resetting Environment & Bots...")
            environment = None; hardcoded_bots = []; learning_bots = []; all_bots = []
            try:
                 # Pass only necessary factors for env generation
                 config_factors = { 'MIN_GOAL_START_DISTANCE_FACTOR': CONFIG['MIN_GOAL_START_DISTANCE_FACTOR'], 'MIN_BOT_START_DISTANCE_FACTOR': CONFIG['MIN_BOT_START_DISTANCE_FACTOR'], 'MIN_BOT_GOAL_DISTANCE_FACTOR': CONFIG['MIN_BOT_GOAL_DISTANCE_FACTOR'] }
                 environment = GridEnvironment(CONFIG['GRID_SIZE'], CONFIG['NUM_GOALS'], CONFIG['MIN_OBSTACLES_FACTOR'], CONFIG['MAX_OBSTACLES_FACTOR'], CONFIG['NUM_HC_BOTS'], CONFIG['NUM_LEARNING_BOTS'], config_factors)
            except Exception as error:
                 print(f"Failed to create Environment: {error}"); traceback.print_exc()
                 await send_status_update(f"Error: Failed Env init! {error}", is_error=True); return False

            total_bots_required = CONFIG['NUM_HC_BOTS'] + CONFIG['NUM_LEARNING_BOTS']
            if not environment.start_positions or len(environment.start_positions) != total_bots_required:
                print(f"Error: Bot/StartPos count mismatch! Req={total_bots_required}, Found={len(environment.start_positions)}"); await send_status_update("Error: Bot start pos mismatch!", is_error=True); return False

            # Create bots
            bot_creation_error = False
            for start_info in environment.start_positions:
                 try:
                     if start_info['type'] == 'Hardcoded':
                          hardcoded_bots.append(HardcodedBot(start_info, start_info['id']))
                     elif start_info['type'] == 'Learning':
                          if not cortical_library: raise ValueError("Library missing for LearningBot.")
                          learning_bots.append(LearningBot(start_info, start_info['id'], cortical_library, CONFIG)) # Pass global CONFIG
                 except Exception as bot_error:
                      print(f"Failed to create bot {start_info['id']}: {bot_error}"); traceback.print_exc()
                      await send_status_update(f"Error: Failed Bot init! {bot_error}", is_error=True); bot_creation_error = True; break
            if bot_creation_error: return False

            all_bots = [*hardcoded_bots, *learning_bots]

            # Update round number
            if not perform_full_reset: round_number = max(1, round_number + 1) # Increment if only round reset
            else: round_number = 1 # Reset to 1 if full reset occurred

            # Reset all hidden states in the library for the new round/bots
            if cortical_library: cortical_library.reset_all_hidden_states()
            print("Environment and Bots reset/created.")

        # Ensure bots are reset even if round wasn't fully reset (e.g., Stop -> Start)
        elif environment: # If only starting, but env exists from previous run
            print("Resetting existing bots for new run...")
            for bot in all_bots:
                 start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                 if start_info: bot.reset(start_info)
                 else: print(f"Warning: No start pos for {bot.id} during soft reset."); bot.reset({'x':1,'y':1}) # Fallback
            if cortical_library: cortical_library.reset_all_hidden_states() # Also reset hidden states here

        # --- Send Initial State to Clients ---
        if environment: await broadcast_message({"type": "environment_update", "payload": environment.get_state_for_client()})
        await broadcast_message({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update()
        await send_library_state_update()
        await send_config_update() # Send final config state
        await send_status_update("Status: Ready")
        print("Setup complete.")
        return True


async def reset_round_state_internal(randomize_env: bool = True):
    """Internal helper to reset the round without full library/score reset."""
    global round_number
    if not environment or not all_bots:
        print("Error: Cannot reset round state - components missing."); await send_status_update("Error: Components missing!", is_error=True); return False

    async with _simulation_lock:
        stop_simulation_internal(update_ui=False) # Ensure loop is stopped

        if randomize_env:
            round_number += 1; print(f"--- Starting Round {round_number} ---"); await send_status_update(f"Status: Starting Round {round_number}...")
            try:
                 environment.randomize() # Regenerate obstacles, goals, start positions
            except Exception as error:
                 print(f"Error randomizing environment: {error}"); traceback.print_exc(); await send_status_update("Error randomizing env!", is_error=True); return False

            # Validate start positions after randomize
            total_bots_required = len(all_bots)
            if not environment.start_positions or len(environment.start_positions) != total_bots_required:
                print(f"Error: Start pos mismatch post-randomize! Req={total_bots_required}, Found={len(environment.start_positions)}"); await send_status_update("Error: Bot count mismatch after randomize!", is_error=True); return False

            # Reset bots with new positions
            for bot in all_bots:
                start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                if start_info: bot.reset(start_info)
                else: print(f"Error: No start pos for bot {bot.id} after randomize!"); bot.reset({'x':1,'y':1}) # Fallback
        else:
            # Just reset claimed goals and bot states/positions to original starts
            print("Resetting round state (keeping layout)...")
            environment.claimed_goals.clear()
            for bot in all_bots:
                start_info = next((sp for sp in environment.start_positions if sp['id'] == bot.id), None)
                if start_info: bot.reset(start_info)
                else: print(f"Error: No start pos for bot {bot.id} during state reset!"); bot.reset({'x':1,'y':1}) # Fallback

        # Reset library hidden states for all bots in either case
        if cortical_library: cortical_library.reset_all_hidden_states()

        # Send updates
        await broadcast_message({"type": "environment_update", "payload": environment.get_state_for_client()})
        await broadcast_message({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update(); await send_library_state_update()
        await send_status_update("Status: Ready") # Set status back to Ready
        return True

# --- Main Simulation Loop Task ---
async def simulation_loop():
    global is_running, environment, all_bots, hc_total_goals, learning_total_goals
    print("Simulation loop started.")
    step_counter = 0
    error_occurred = False # Track if any error stops the loop

    while is_running:
        start_time = time.monotonic()
        round_over_signal = None
        current_step_error = False
        bots_state_update = []
        goals_claimed_this_step = False
        step_losses = []

        async with _simulation_lock:
            # --- Pre-step Checks ---
            if not is_running or not environment or not all_bots or not cortical_library:
                 print("Simulation loop stopping mid-step: Critical components missing.");
                 is_running = False; error_occurred = True; break

            # --- 1. Get Actions & Senses ---
            bot_actions: Dict[str, int] = {}
            bot_senses_current: Dict[str, Dict] = {} # Senses BEFORE action
            for bot in all_bots:
                if bot.steps >= CONFIG['MAX_STEPS_PER_ROUND']: continue # Skip finished bots
                action = -1; senses = None
                try:
                    senses = environment.get_sensory_data(bot, all_bots, CONFIG['VISIBILITY_RANGE'])
                    bot_senses_current[bot.id] = senses
                    action = bot.get_action(senses, environment) # This calls library predict for LearningBots
                    bot_actions[bot.id] = action
                except Exception as error:
                    print(f"Error getting action/senses for bot {bot.id}: {error}"); traceback.print_exc()
                    await send_status_update(f"Error processing {bot.id}! Check server console.", is_error=True);
                    current_step_error = True; error_occurred = True; break
            if current_step_error: is_running = False; break

            # --- 2. Resolve Actions & Update State ---
            punched_bots: Set[str] = set()
            bot_next_pos: Dict[str, Dict] = {}
            # Calculate intended positions and identify punches
            for bot in all_bots:
                action = bot_actions.get(bot.id, -1)
                current_pos = bot.pos.copy(); next_p = current_pos
                if action != -1 and bot.freeze_timer <= 0: # Only act if not frozen and action exists
                    if 0 <= action <= 3: # Move
                        intended_pos = environment.perform_move_action(current_pos, action)
                        if environment.is_valid(intended_pos): next_p = intended_pos
                    elif action == 4: # Punch
                        target_bot = next((ob for ob in all_bots if ob.id != bot.id and manhattan_distance(current_pos, ob.pos) == 1 and ob.freeze_timer <= 0), None)
                        if target_bot: punched_bots.add(target_bot.id)
                    # Action 5 (Claim) doesn't change position here, handled after update
                bot_next_pos[bot.id] = next_p # Store intended/current pos

            # Apply state updates (freeze, position, goals)
            for bot in all_bots:
                 next_pos = bot_next_pos.get(bot.id, bot.pos)
                 action = bot_actions.get(bot.id, -1)
                 if bot.id in punched_bots: bot.apply_freeze(CONFIG['FREEZE_DURATION'])
                 bot.update(next_pos, action) # Updates pos, steps, internal freeze timer

                 # Handle goal claiming *after* position update
                 if action == 5 and bot.freeze_timer <= 0:
                      adjacent_goal = environment.get_adjacent_unclaimed_goal(bot.pos)
                      if adjacent_goal and environment.claim_goal(adjacent_goal['id'], bot.id):
                            bot.goals_reached_this_round += 1
                            if bot.type == 'Hardcoded': hc_total_goals += 1
                            else: learning_total_goals += 1
                            goals_claimed_this_step = True
                            if environment.are_all_goals_claimed():
                                round_over_signal = 'goals_claimed'; print(f"--- Final goal {adjacent_goal['id']} claimed by {bot.id}! ---")

                 bots_state_update.append(bot.get_state_for_client())

            # --- 3. Learning Step (After all bots moved/acted) ---
            bot_senses_next: Dict[str, Dict] = {} # Senses AFTER action
            # Only learn if no errors, round isn't over yet, and not first step
            if not current_step_error and not round_over_signal and step_counter > 0:
                # Get senses for all bots in their *new* positions
                for bot in all_bots:
                    try: bot_senses_next[bot.id] = environment.get_sensory_data(bot, all_bots, CONFIG['VISIBILITY_RANGE'])
                    except Exception as sense_error:
                         print(f"Error getting next senses for {bot.id}: {sense_error}"); current_step_error = True; error_occurred = True; break
                if not current_step_error:
                    for bot in learning_bots:
                         if bot.id in bot_senses_next and bot.last_prediction_vector is not None: # Can only train if prediction exists
                              try:
                                  loss = bot.train_on_last_step(bot_senses_next[bot.id])
                                  if loss is not None and loss >= 0: step_losses.append(loss)
                              except Exception as train_error: print(f"Error during training for {bot.id}: {train_error}"); traceback.print_exc()
            if current_step_error: is_running = False; break

            # --- 4. Send Updates ---
            if goals_claimed_this_step: await broadcast_message({"type": "environment_update", "payload": environment.get_state_for_client()})
            if bots_state_update: await broadcast_message({"type": "bots_update", "payload": bots_state_update})
            if step_counter % 5 == 0 or goals_claimed_this_step: # Update stats/library periodically or on goal claim
                 await send_stats_update()
                 await send_library_state_update()

            # --- 5. Check Round End Conditions ---
            if not round_over_signal:
                 all_bots_done = not all_bots or all(b.steps >= CONFIG['MAX_STEPS_PER_ROUND'] for b in all_bots)
                 if all_bots_done: round_over_signal = 'max_steps'
                 # Final check on goals after all updates
                 elif environment.are_all_goals_claimed(): round_over_signal = 'goals_claimed'

            step_counter += 1
        # --- End of _simulation_lock for the step ---

        # --- Handle Round End and Reset (Outside Lock) ---
        if round_over_signal:
            reason = "All goals claimed" if round_over_signal == 'goals_claimed' else "Max steps reached"
            print(f"--- End of Round {round_number} ({reason}) ---")
            if is_running: # Auto-start next round if still running
                try:
                    if not await reset_round_state_internal(randomize_env=True):
                        print("Reset round failed. Stopping simulation."); await send_status_update("Error: Failed to reset round. Stopped.", is_error=True); is_running = False
                    else: step_counter = 0 # Reset step counter for new round
                except Exception as reset_error:
                    print(f"Error auto-resetting round: {reset_error}"); traceback.print_exc(); await send_status_update("Error resetting round! Stopped.", is_error=True); is_running = False
            else: # Loop was stopped externally (e.g., user click, error)
                 await send_status_update(f"Round {round_number} Finished ({reason}). Stopped.")

        # --- Delay ---
        if is_running:
            end_time = time.monotonic(); elapsed_ms = (end_time - start_time) * 1000
            delay_s = max(0.001, (CONFIG['SIMULATION_SPEED_MS'] - elapsed_ms) / 1000.0)
            await asyncio.sleep(delay_s)

    # --- Loop End ---
    print("Simulation loop finished.")
    try:
        if environment: await broadcast_message({"type": "environment_update", "payload": environment.get_state_for_client()})
        if all_bots: await broadcast_message({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]})
        await send_stats_update()
    except Exception as final_update_e: print(f"Error sending final updates: {final_update_e}")

    if not error_occurred: await send_status_update("Status: Stopped.")
    # If error occurred, the error status should persist


def start_simulation_internal():
    global simulation_task, is_running
    if is_running or (simulation_task and not simulation_task.done()): return False
    print("Starting simulation task."); is_running = True; simulation_task = asyncio.create_task(simulation_loop()); return True

def stop_simulation_internal(update_ui=True):
    global is_running, simulation_task
    if not is_running and (not simulation_task or simulation_task.done()): return
    print("Stopping simulation task..."); is_running = False
    # Don't await task here, just signal stop. Loop handles exit.
    # if update_ui: asyncio.create_task(send_status_update("Status: Stopping..."))


# ================================================================
# --- FastAPI App and WebSocket Endpoint ---
# ================================================================
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global CONFIG, is_running, simulation_task
    await manager.connect(websocket)
    print(f"Client connected: {websocket.client}")

    try:
        # Send initial state on connect
        await websocket.send_text(json.dumps({"type": "config_update", "payload": CONFIG}))
        async with _simulation_lock: # Access shared state safely
            if environment: await websocket.send_text(json.dumps({"type": "environment_update", "payload": environment.get_state_for_client()}))
            if all_bots: await websocket.send_text(json.dumps({"type": "bots_update", "payload": [bot.get_state_for_client() for bot in all_bots]}))
            # Send global stats/library state via broadcast helpers which handle locking/sending
            await send_stats_update()
            await send_library_state_update()
            status_msg = "Status: Running" if is_running else ("Status: Ready" if environment else "Status: Initializing")
            await websocket.send_text(json.dumps({"type": "status_update", "payload": {"message": status_msg}}))


        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            payload = message.get("payload", {})

            if msg_type == "start":
                async with _simulation_lock: is_ready = bool(environment and all_bots)
                if not is_running and is_ready:
                     if start_simulation_internal(): await send_status_update("Status: Running...")
                     else: await send_status_update("Status: Start Failed.", is_error=True)
                elif is_running: await websocket.send_text(json.dumps({"type": "status_update", "payload": {"message": "Status: Already Running", "is_error": False}}))
                else: await websocket.send_text(json.dumps({"type": "status_update", "payload": {"message": "Status: Not Ready, Reset First", "is_error": True}}))

            elif msg_type == "stop":
                 if is_running: stop_simulation_internal(update_ui=False) # Signal stop, loop handles UI update

            elif msg_type == "resetRound":
                 await reset_round_state_internal(randomize_env=True) # Handles stop and UI updates

            elif msg_type == "resetAll":
                 # Force full reset, library compatibility doesn't matter here
                 await setup_simulation(is_full_reset_request=True, loaded_library_compatible=True)

            elif msg_type == "updateParam":
                 key = payload.get("key"); value = payload.get("value")
                 if key and key in CONFIG:
                      current_type = type(DEFAULT_CONFIG.get(key))
                      new_value = None
                      try:
                          if current_type == int: new_value = int(value)
                          elif current_type == float: new_value = float(value)
                          elif current_type == bool: new_value = bool(value)
                          elif isinstance(DEFAULT_CONFIG.get(key), list):
                               # Basic check for list type (e.g., FEATURE flags)
                               if isinstance(value, list): new_value = value # Assume valid list for now
                               else: raise TypeError("Expected a list")
                          else: new_value = value # Assume string or other

                          if new_value is not None and type(new_value) == current_type:
                              temp_config = CONFIG.copy(); temp_config[key] = new_value
                              needs_full, needs_round = check_config_needs_reset(temp_config)
                              is_live_update = not needs_full and not needs_round

                              if is_running and not is_live_update:
                                  await websocket.send_text(json.dumps({"type": "action_feedback", "payload": {"success": False, "message": f"Stop simulation to change '{key}'."}}))
                              else:
                                    print(f"Updating config: {key} = {new_value}")
                                    CONFIG[key] = new_value
                                    # Recalculate dependent config if needed
                                    if key.startswith("FEATURE") or key == "NUM_ACTIONS":
                                         CONFIG['LIB_INPUT_VECTOR_SIZE'] = calculate_input_vector_size(CONFIG)
                                         print(f"Recalculated LIB_INPUT_VECTOR_SIZE: {CONFIG['LIB_INPUT_VECTOR_SIZE']}")
                                         needs_full = True # Feature changes always need full reset

                                    await send_config_update() # Broadcast updated config
                                    if not is_running:
                                        if needs_full: await send_status_update(f"Status: Config updated. 'Reset All' required.")
                                        elif needs_round: await send_status_update(f"Status: Config updated. 'New Round' required.")
                          else: print(f"Type mismatch or conversion failed for param {key}. Expected {current_type}, got {type(value)}")
                      except (ValueError, TypeError) as e: print(f"Invalid value type for param {key}: {value} - {e}")

            elif msg_type == "saveParams":
                 success_save = save_params_to_file()
                 success_lib_save = False
                 if cortical_library:
                     try: cortical_library.save_state(); success_lib_save = True
                     except Exception as e: print(f"Error saving library state: {e}")
                 msg = f"{'Params saved. ' if success_save else 'Param save failed. '}{'Library state saved.' if success_lib_save else 'Library save failed.'}"
                 await websocket.send_text(json.dumps({"type": "action_feedback", "payload": {"success": success_save and success_lib_save, "message": msg}}))

            elif msg_type == "loadParams":
                 stop_simulation_internal(update_ui=False)
                 params_loaded_ok, lib_compatible = load_params_from_file()
                 setup_ok = False
                 if params_loaded_ok:
                     # Force full reset after loading params, pass compatibility flag
                     setup_ok = await setup_simulation(is_full_reset_request=True, loaded_library_compatible=lib_compatible)
                     msg = "Parameters loaded."
                     if setup_ok: msg += " Reset performed." + (" Library loaded from state." if lib_compatible and os.path.exists("cortical_library_state.pth") else " Library initialized fresh.")
                     else: msg += " Simulation setup FAILED after load."
                     await websocket.send_text(json.dumps({"type": "action_feedback", "payload": {"success": params_loaded_ok and setup_ok, "message": msg}}))
                 else:
                     await websocket.send_text(json.dumps({"type": "action_feedback", "payload": {"success": False, "message": "Error loading parameters file. Using defaults."}}))
                     await setup_simulation(is_full_reset_request=True, loaded_library_compatible=False) # Reset with defaults

            elif msg_type == "resetParams":
                 stop_simulation_internal(update_ui=False)
                 reset_params_to_default()
                 lib_state_file = "cortical_library_state.pth"
                 deleted_state = False
                 if os.path.exists(lib_state_file):
                     try: os.remove(lib_state_file); deleted_state = True; print("Deleted saved library state file.")
                     except OSError as e: print(f"Error deleting saved library state: {e}")
                 # Reset requires full setup with fresh library
                 await setup_simulation(is_full_reset_request=True, loaded_library_compatible=False)
                 msg = "Parameters reset to default." + (" Saved library state deleted." if deleted_state else "") + " Reset performed."
                 await websocket.send_text(json.dumps({"type": "action_feedback", "payload": {"success": True, "message": msg}}))

    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket Error: {e}"); traceback.print_exc()
    finally:
        manager.disconnect(websocket)
        print(f"Client connection closed: {websocket.client}. Remaining: {len(manager.active_connections)}")


@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    global CONFIG, cortical_library
    # Load params first, check compatibility
    _, lib_compatible = load_params_from_file()
    # Setup simulation, attempt library load if compatible
    await setup_simulation(is_full_reset_request=True, loaded_library_compatible=lib_compatible)
    if not cortical_library:
         print("\n" + "="*60)
         print("CRITICAL ERROR: Learning library failed to initialize on startup.")
         print("The simulation will not function correctly.")
         print("Check GPU drivers, PyTorch installation, and config.")
         print("="*60 + "\n")


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
        try: cortical_library.dispose()
        except Exception as e: print(f"Error disposing library on shutdown: {e}")
    print("Shutdown complete.")

# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def get_index():
    script_dir = os.path.dirname(__file__); file_path = os.path.join(script_dir, "index.html")
    if os.path.exists(file_path):
         with open(file_path, "r") as f: return HTMLResponse(content=f.read())
    else: return HTMLResponse(content="<h1>Error: index.html not found!</h1> Check if it's in the same directory as server.py.", status_code=404)

if __name__ == "__main__":
    if CorticalLearningLibrary is None:
         print("Learning library failed to import. Server cannot start.")
    else:
        print("Starting Multi-Bot GPU Learning Server (v6 - External Library)...")
        # Get host/port from environment variables or use defaults
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        print(f"Server will listen on {host}:{port}")
        uvicorn.run(app, host=host, port=port)