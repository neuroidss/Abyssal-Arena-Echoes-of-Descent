# Multi-Bot Titans Simulation (v5.7)

## Overview

This project simulates a multi-agent environment where different types of bots compete to claim goals while navigating obstacles and interacting with each other (punching to freeze). It features:

*   **Hardcoded Bots:** Operate based on predefined rules (pathfinding, basic interaction logic).
*   **Learning Bots:** Utilize a backend PyTorch implementation inspired by the "Titans: Learning to Memorize at Test Time" paper. They learn online based on prediction anomalies within their internal Neural Memory module.
*   **Player Control:** Users can join the simulation and take control of an available Learning Bot via a web interface, setting targets by clicking/tapping the grid.
*   **GPU Acceleration:** The PyTorch backend leverages CUDA GPUs if available for accelerating the Learning Bots' neural network computations.
*   **Web Interface:** A Flask+SocketIO server provides a real-time web interface (index.html) for visualization, control, and parameter tuning.

The core idea is to simulate a system where "intelligent" agents (Learning Bots) develop strategies based purely on processing input streams (sensors + actions) and learning from prediction errors within their memory, mimicking concepts from cortical learning theories.

## Features (v5.7)

*   **Titans-Inspired Learning:** Learning Bots use a `NeuralMemoryManager` in PyTorch, performing online learning based on prediction loss (anomaly).
*   **Player Target Following:** Click/Tap the grid to set a target coordinate for your controlled bot. The bot will attempt to move towards, punch opponents at, or claim goals at the target location. Tap the bot itself to cancel the target.
*   **Player Join/Leave/Rejoin:**
    *   Join by clicking an available (non-player-controlled) Learning Bot on the canvas when the simulation is stopped.
    *   Leave control using the "Leave Bot" button.
    *   Automatic rejoin attempt on page reload if previously controlling a bot (uses browser localStorage).
*   **Improved Hardcoded Bot Logic:** Reduced instances of bots getting stuck near walls or other bots.
*   **Dynamic Parameter Tuning:** Adjust simulation speed, bot counts, grid size, learning parameters, etc., via the UI. Some changes require a round reset or full reset.
*   **Save/Load UI Parameters:** Persist UI parameter settings in the browser's localStorage.
*   **Real-time Visualization:** Canvas shows bot positions, goals, obstacles, player targets, and bot status (frozen, player-controlled).
*   **Mobile Controls:** Basic directional and action buttons for player control on touch devices (though tap-to-target is the primary method).
*   **GPU Acceleration:** Automatically uses CUDA if detected by PyTorch.

## Setup and Running

**Prerequisites:**

1.  **Python 3.8+**
2.  **PyTorch:** Install with CUDA support if you have a compatible NVIDIA GPU. Visit [https://pytorch.org/](https://pytorch.org/) for instructions. Verify installation with `python -c "import torch; print(torch.cuda.is_available())"`.
3.  **Flask, Flask-SocketIO, Eventlet:**
    ```bash
    pip install Flask Flask-SocketIO eventlet numpy
    ```

**Running the Simulation:**

1.  Save the provided code blocks as `server.py` and `index.html` in the same directory.
2.  Open a terminal or command prompt in that directory.
3.  Run the server:
    ```bash
    python server.py
    ```
4.  The server will start (usually on `http://0.0.0.0:5001`). Check the console output for the exact address and CUDA status.
5.  Open a web browser and navigate to the address printed by the server (e.g., `http://localhost:5001` or `http://<your-server-ip>:5001`).
6.  Use the controls in the web interface:
    *   Adjust parameters (Apply/Reset as needed).
    *   Click "Start" to begin the simulation.
    *   Click "Stop" to pause.
    *   Click an available Learning Bot (orange dashed outline when stopped) to join as a player.
    *   Click/Tap the grid to set a target for your bot.
    *   Click "Leave Bot" to return control to the AI.

## Key Concepts Implemented

*   **Neural Memory (Titans-Inspired):** The `NeuralMemoryManager` class manages individual bot memory states. Each step involves retrieval (inference) and an update based on prediction error (MSE loss between the memory's output for the input and a target derived from the input). Learning happens online at "test time".
*   **Stream Agnosticism:** The `NeuralMemoryManager` processes a single input tensor (`_get_input_tensor_for_bot`) combining sensory data and the last action. It doesn't inherently know which part is a sensor and which is an action.
*   **Anomaly-Based Learning:** The MSE loss calculated during the memory update serves as a proxy for prediction anomaly, driving the learning process via standard backpropagation and AdamW optimization within the `forward_step`.
*   **Embodied AI:** Bots perceive the environment (`get_sensory_data`), decide on an action (`get_hardcoded_action`, `get_learning_action`), and execute it, affecting the environment and their future perceptions.
*   **Player Interaction:** Allows direct human input to guide a learning bot, potentially providing diverse data for the learning process (though the learning is still unsupervised based on internal prediction error).

## Code Structure (`server.py`)

*   **Flask/SocketIO Setup:** Handles web server and real-time communication.
*   **PyTorch Setup:** Detects and configures the device (CPU/GPU).
*   **NeuralMemory Implementation:** Contains the `LayerNorm`, `MemoryMLP`, `NeuralMemState`, and `NeuralMemoryManager` classes adapted for the simulation.
*   **Default Configuration:** Defines tunable parameters.
*   **Global State:** Manages `bots`, `players`, `environment`, `neural_memory_manager`, etc.
*   **GridEnvironment Class:** Manages the grid, goals, obstacles, pathfinding, and sensory data generation.
*   **Bot Logic Functions:** `get_hardcoded_action`, `_get_input_tensor_for_bot`, `get_learning_action`.
*   **Simulation Setup & Control:** `create_..._bot_instance`, `setup_simulation`, `get_game_state`, `simulation_step`, `simulation_loop`, `emit_state`.
*   **SocketIO Event Handlers:** Manage client connections, disconnections, game actions (start, stop, reset), parameter updates, and player control (join, leave, target updates, direct actions).
*   **Server Start:** Initializes everything and runs the Flask-SocketIO app using eventlet.

## Future Directions / TODO

*   Implement more sophisticated memory architectures (e.g., attention-based memory from Titans).
*   Explore hierarchical memory structures (stacking `NeuralMemory` layers).
*   Visualize internal memory states (SDRs, weights - complex).
*   Implement more advanced exploration strategies based on anomaly levels.
*   Refine bot interaction behaviors.
*   Optimize performance for very large grids/bot counts.

