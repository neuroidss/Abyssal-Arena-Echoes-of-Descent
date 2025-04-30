# learning_library.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import traceback
from typing import Dict, Any, Tuple, Optional, List

class CorticalLearningLibrary:
    """
    GPU-Accelerated Agnostic Sequence Learning Library.

    Processes numerical input streams, predicts the next element in the stream,
    and learns based on the prediction error (loss/anomaly).

    Manages separate hidden states for multiple independent sequences (bots).
    Uses a single shared model on the specified device (GPU/CPU).
    """
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._determine_device(config.get('LIB_DEVICE', 'auto'))
        print(f"Initializing Cortical Learning Library on device: {self.device}")

        # --- Stream Configuration (Passed from Server) ---
        self.input_vector_size = config.get('LIB_INPUT_VECTOR_SIZE')
        if not self.input_vector_size:
            raise ValueError("LIB_INPUT_VECTOR_SIZE must be specified in config.")
        self.output_vector_size = self.input_vector_size # Predicts the next input vector

        # --- PyTorch Model Definition ---
        self.rnn_hidden_size = config.get('LIB_HIDDEN_SIZE', 512) # Larger default
        self.rnn_num_layers = config.get('LIB_NUM_LAYERS', 3)    # More layers default
        self.rnn_dropout = config.get('LIB_DROPOUT', 0.1)

        # No embeddings needed here - assumes pre-encoded numerical vectors
        print(f"Library Input Size: {self.input_vector_size}, Output Size: {self.output_vector_size}")

        # RNN Layer
        rnn_class = nn.LSTM if config.get('LIB_RNN_TYPE', 'LSTM') == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            input_size=self.input_vector_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True, # Expect input shape (batch, seq, feature)
            dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0
        ).to(self.device)

        # Output Layer to predict the *next* input vector
        self.output_layer = nn.Linear(self.rnn_hidden_size, self.output_vector_size).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.get('LIB_LEARNING_RATE', 0.0005)) # Smaller LR default

        # Loss Function
        loss_type = config.get('LIB_LOSS_TYPE', 'MSE')
        if loss_type == "MSE": self.criterion = nn.MSELoss()
        elif loss_type == "L1": self.criterion = nn.L1Loss()
        elif loss_type == "Huber": self.criterion = nn.HuberLoss()
        else:
            print(f"Warning: Unknown loss type '{loss_type}'. Defaulting to MSE.")
            self.criterion = nn.MSELoss()

        # Store hidden states per bot instance {bot_id: hidden_state_tuple/tensor}
        self.hidden_states: Dict[str, Any] = {}

        # Track loss
        self.loss_history = []
        self.max_loss_history = 100 # Keep track of recent losses

    def _determine_device(self, requested_device: str) -> torch.device:
        if requested_device == "cuda":
            if torch.cuda.is_available(): return torch.device("cuda")
            else: print("Warning: CUDA requested but not available. Using CPU."); return torch.device("cpu")
        elif requested_device == "cpu":
            return torch.device("cpu")
        else: # auto
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parameters(self): # Helper to get all model parameters for optimizer
        return list(self.rnn.parameters()) + list(self.output_layer.parameters())

    def get_initial_hidden_state(self) -> Any:
         """Returns a new zeroed hidden state for a single sequence (batch=1)."""
         batch_size = 1 # Each bot processes its sequence individually
         h = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=self.device)
         if self.config.get('LIB_RNN_TYPE', 'LSTM') == 'LSTM':
             c = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=self.device)
             return (h, c)
         else: # GRU
             return h

    def get_persistent_hidden_state(self, bot_id: str) -> Any:
        """Retrieves the stored hidden state for a bot, or creates a new one."""
        if bot_id not in self.hidden_states:
            self.hidden_states[bot_id] = self.get_initial_hidden_state()
        return self.hidden_states[bot_id]

    def store_hidden_state(self, bot_id: str, state: Any):
        """Stores the updated hidden state for a bot, detaching it from the graph."""
        if isinstance(state, tuple): # LSTM state (h, c)
            # Detach AND clone to ensure no lingering graph references
            self.hidden_states[bot_id] = (state[0].detach().clone(), state[1].detach().clone())
        else: # GRU state (h)
            self.hidden_states[bot_id] = state.detach().clone()

    def reset_hidden_state(self, bot_id: str):
         """Resets the hidden state for a specific bot."""
         if bot_id in self.hidden_states:
             del self.hidden_states[bot_id] # Remove it, will be recreated on next access

    def reset_all_hidden_states(self):
        """Resets hidden states for all tracked bots."""
        self.hidden_states.clear()

    def process_and_predict(self, bot_id: str, input_vector: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Processes a single input vector for a given bot_id and predicts the next vector.

        Args:
            bot_id: The unique identifier for the sequence/bot.
            input_vector: A NumPy array representing the current input state.

        Returns:
            A tuple containing:
            - The predicted next input vector as a NumPy array (or None if error).
            - An estimated 'confidence' or inverse anomaly score (currently placeholder 1.0).
              A lower value would indicate higher anomaly/lower confidence.
              Actual anomaly calculation requires comparing with the *next actual* input.
        """
        if input_vector.size != self.input_vector_size:
            print(f"Error: Input vector size mismatch for {bot_id}. Expected {self.input_vector_size}, got {input_vector.size}")
            return None, 0.0

        # 1. Prepare Input Tensor
        input_tensor = torch.tensor(input_vector, dtype=torch.float32, device=self.device)
        # Reshape to (batch=1, seq=1, features) for RNN
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        # 2. Get Hidden State
        hidden_state = self.get_persistent_hidden_state(bot_id)

        # 3. Run Prediction
        self.rnn.eval() # Set model to evaluation mode
        prediction_tensor_next = None
        next_hidden_state = None
        try:
            with torch.no_grad(): # No need to track gradients for prediction
                rnn_output, next_hidden_state_raw = self.rnn(input_tensor, hidden_state)
                # rnn_output shape: (batch=1, seq=1, hidden_size)
                prediction_tensor_next = self.output_layer(rnn_output.squeeze(1)) # Squeeze seq dim
                # prediction shape: (batch=1, output_features)

            # 4. Store updated hidden state (must be done *after* using the old one)
            self.store_hidden_state(bot_id, next_hidden_state_raw)

            # 5. Convert prediction to NumPy array for output
            prediction_np = prediction_tensor_next.squeeze(0).detach().cpu().numpy()
            return prediction_np, 1.0 # Placeholder confidence

        except Exception as e:
            print(f"Error during prediction for {bot_id}: {e}")
            traceback.print_exc()
            # If prediction fails, don't update hidden state? Or reset it?
            # For now, we just return None, hidden state remains as it was before the call.
            return None, 0.0

    def train_on_step(self, prediction_vector: np.ndarray, actual_next_vector: np.ndarray) -> float:
        """
        Performs a single training step based on a previous prediction and the actual outcome.

        Args:
            prediction_vector: The NumPy array predicted in the previous step.
            actual_next_vector: The NumPy array representing the actual state that occurred.

        Returns:
            The calculated loss for this step (float). Returns -1.0 if error.
        """
        if prediction_vector.size != self.output_vector_size or actual_next_vector.size != self.input_vector_size:
             print(f"Error: Training vector size mismatch. Pred={prediction_vector.size}, Actual={actual_next_vector.size}")
             return -1.0 # Indicate error

        # 1. Prepare Tensors
        # Prediction tensor needs requires_grad_() for backprop
        prediction_tensor = torch.tensor(prediction_vector, dtype=torch.float32, device=self.device).requires_grad_(True)
        target_tensor = torch.tensor(actual_next_vector, dtype=torch.float32, device=self.device)

        # Ensure tensors have the correct shape if criterion expects specific dims
        # Assuming criterion works on vectors (N) or (1, N)
        if prediction_tensor.dim() == 1: prediction_tensor = prediction_tensor.unsqueeze(0)
        if target_tensor.dim() == 1: target_tensor = target_tensor.unsqueeze(0)

        # 2. Perform Training Step
        self.rnn.train() # Set model to training mode
        self.optimizer.zero_grad()

        try:
            loss = self.criterion(prediction_tensor, target_tensor)

            if torch.isnan(loss):
                print("Warning: Loss is NaN. Skipping backpropagation.")
                # Log tensors that caused NaN?
                # print("Pred:", prediction_tensor)
                # print("Target:", target_tensor)
                return 0.0 # Return 0 loss if NaN to avoid polluting history

            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track loss
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            if len(self.loss_history) > self.max_loss_history:
                self.loss_history.pop(0)

            return current_loss

        except Exception as e:
            print(f"Error during training step: {e}")
            traceback.print_exc()
            return -1.0 # Indicate error


    def get_average_loss(self) -> Optional[float]:
        """Calculates the average loss over the stored history."""
        if not self.loss_history: return None
        # Filter out potential -1.0 error indicators if necessary
        valid_losses = [l for l in self.loss_history if l >= 0]
        if not valid_losses: return None
        return sum(valid_losses) / len(valid_losses)

    def get_internal_state_info(self) -> Dict:
         """Provides diagnostic information about the library."""
         avg_loss = self.get_average_loss()
         return {
             'device': str(self.device),
             'rnnType': self.config.get('LIB_RNN_TYPE', 'LSTM'),
             'hiddenSize': self.rnn_hidden_size,
             'numLayers': self.rnn_num_layers,
             'inputVectorSize': self.input_vector_size,
             'avgLoss': f"{avg_loss:.5f}" if avg_loss is not None else "N/A",
             'lossHistory': ", ".join(f"{l:.4f}" for l in self.loss_history[-15:]), # Last 15 loss points
             'trackedBots': len(self.hidden_states) # Number of bots with active hidden states
         }

    def save_state(self, path: str = "cortical_library_state.pth"):
        """Saves the model, optimizer state, and config."""
        state = {
            'config': self.config, # Save config used to create the model
            'model_state_dict': {
                'rnn': self.rnn.state_dict(),
                'output_layer': self.output_layer.state_dict(),
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            # Note: Hidden states are transient and typically not saved/loaded
        }
        try:
            torch.save(state, path)
            print(f"Library state saved to {path}")
        except Exception as e:
            print(f"Error saving library state: {e}")

    def load_state(self, path: str = "cortical_library_state.pth"):
        """Loads the model and optimizer state, checking for compatibility."""
        if not os.path.exists(path):
            print(f"Info: Library state file not found at {path}. Starting fresh.")
            return False
        try:
            state = torch.load(path, map_location=self.device) # Load directly to the correct device

            # --- Compatibility Check ---
            loaded_config = state.get('config', {})
            # Critical structure params:
            structure_keys = [
                'LIB_RNN_TYPE', 'LIB_HIDDEN_SIZE', 'LIB_NUM_LAYERS',
                'LIB_INPUT_VECTOR_SIZE' # Must match!
            ]
            mismatch = False
            for key in structure_keys:
                # Need to handle cases where key might not exist in old save or current default
                loaded_val = loaded_config.get(key)
                current_val = self.config.get(key) # Use .get for safety
                if loaded_val is None or current_val is None or loaded_val != current_val:
                     print(f"Error: Config mismatch on load! Key '{key}'. Saved: {loaded_val}, Current: {current_val}")
                     mismatch = True
            if mismatch:
                 print("Cannot load state due to incompatible model structure. Re-initialize or use matching config.")
                 # Delete incompatible state file? Optional.
                 # try: os.remove(path); print(f"Deleted incompatible state file: {path}")
                 # except OSError as e: print(f"Could not delete incompatible state file: {e}")
                 return False
            # --- End Compatibility Check ---

            model_state = state['model_state_dict']
            self.rnn.load_state_dict(model_state['rnn'])
            self.output_layer.load_state_dict(model_state['output_layer'])

            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.loss_history = state.get('loss_history', [])

            # Update internal config based on loaded one (for non-structural things like LR if needed)
            self.config.update(loaded_config) # Overwrite current config with loaded

            print(f"Library state loaded successfully from {path}")
            # Reset transient states after loading
            self.reset_all_hidden_states()
            return True

        except Exception as e:
            print(f"Error loading library state: {e}")
            traceback.print_exc()
            # If load fails badly, maybe delete the corrupted file?
            return False

    def dispose(self):
        """Releases resources, especially GPU memory if applicable."""
        self.hidden_states.clear()
        self.loss_history = []
        del self.rnn
        del self.output_layer
        del self.optimizer
        del self.criterion
        if self.device == torch.device("cuda"):
             try:
                 import gc
                 gc.collect() # Run Python garbage collection
                 torch.cuda.empty_cache() # Try to release PyTorch's cached memory
                 print("Attempted to clear PyTorch GPU cache.")
             except Exception as e:
                 print(f"Error during GPU cleanup: {e}")
        print(f"Disposed Cortical Learning Library.")