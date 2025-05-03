# Multi-Bot Titans Simulation (v5.7)

## Overview

This project simulates a multi-agent environment where different types of bots compete to claim goals while navigating obstacles and interacting with each other (punching to freeze). It features:

*   **Hardcoded Bots:** Operate based on predefined rules (pathfinding, basic interaction logic).
*   **Learning Bots:** Utilize a backend PyTorch implementation inspired by the "[Titans: Learning to Memorize at Test Time](https://github.com/neuroidss/Abyssal-Arena-Echoes-of-Descent/blob/main/mvp1/Multi-Bot_Titans_Sim_v5.7_-_Target_Follow_%26_Player_Join/2501.00663v1.pdf)" paper. They learn online based on prediction anomalies within their internal Neural Memory module.
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

# vibe-coding prompt
```
first research how to make SDR (Sparse Distributed Representation) in tfjs.
how to make example where some bot with hardcoded goals will be captured via senses and actions into sdr to learn as bottom-up spatiotemporal hierarchy and make bot with intelligent goals which next actions will be top-down predictions, and anomalies in predictions will rule learning process, make no difference between actions and senses, as future use is to feed eeg which has no specific actions and senses but they need to be learned together with actual sensors and actions when actual sensors and actions will be removed and will remain only eeg.

then make demo where will be bot with hardcoded goals, and then when tfjs will learn internal intelligent goals make another bot with intelligent rules and compare how goals achieved by hardcoded bot and intelligent bot. hardcoded bot should have tasks which achievable in reasonable time and reproducible, to compare different generations of tfjs architectures. make several configurations of tfjs bot to make leaderboard. hardcoded bot should not know where exactly his goals but should search it having only senses and actions, it should take reasonable time to reach its goals.

write all in single html file, but inside html separate library code in which will be only data streams on input and predictions on output to use this library separately later so all tfjs only in this library.

make hardcoded bot run infinitely so intelligent bot could learn new strategies every time. so change environment every time but keep seed. intelligent bot needs to learn to reach goals in any new environment. change environment every time as goals reached by  some of bots. show bots statistics how much goals reached by every of them.

make world larger so make goals far enough to reach so untrained intelligent bot will have zero chance to reach goals faster then hardcoded. because now statistics hardcoded/intelligent 328/24, but it seems because of random moves of untrained bot. make bots not start from only one point so intelligent bot will learn to go in any directions. make bots not just compete each other but also able to punch each other to make opponent stop for some time so intelligent bot will learn to attack, defend or avoid opponents. make multiple goals to reach so no need to run to one direction for both bots. and make bots start far from each other to not start fight immediately, but only in process of reaching goals.

make bot continue moving after stun someone. make all important initial parameter tunable possibly without restart of simulation level, and which not possible without restart level will require level restart, and some will require full reset, but try avoid restarts resets to tune parameters, just use resizes and scaling. make visibility of all goals and opponents in visible range every step and choose which easier to reach, and attack opponent it stops from reaching goals or no goals visible around. stunning stops from acting, making unable to act, but make freezing which make possible to act but all acts will not lead to no result. so after stunning bot will see that it was away for some steps and has no sense for that time, but in freezing bot will see that actions not working and it is it freeze state, so make it feel freezing and its consequences.

make all not stopped after first round. make restart level works, make start simulation works, so simulation not frozen after first round. and make after froze bot not stay on same place freezing other bot until time ends. so make bot see that it freees other bot.

make hardcoded bot not freezes when see goal and there obstacle between bot and goal. make hardcoded bot somehow try to avoid obstacles between it and goals.

make it all vizible okay for mobile device.
made 1ms delay work fine.

make grid size, num goals make real effect, as now goals number not changed really, and seems grid size not make bot act inside new range and obstacles and goals also not placed inside new grid size. so somehow arena size stays same and bots operating inside old grid size with old number of goals after changed and applied force new round, only grid size changed and it empty and unexplorable for bots at new area.

make changeable of bots hardcoded and intelligent, make intelligent bots use same model at first time, but just make ability to choose number of both type of bots.

make it required to make action interaction with goal, not just touch, as with punch of opponent requires action. so intelligent bot will learn maybe how to interact with goals.

make parameters of model and game setup save and restore from localstorage.
make model parameters be changeable to increase hierarchy and other sizes to maximum on current hardware to make most intellegent bots. when localstorage parameters not fit somehow jast turn it to default, also make switch to default.

make inside library absolutely no difference what are senses what are actions. actions are only predictions which intelligent bots used to act outside of library. library only sense and predict all senses, library never act.

use just streams inside of library, absolutely no use of senses or actions inside library, only bots outside library knows what send as actions and what use as actions.

number of bots and grid and goals resize changeable only at reset all.

on start of library need to be only sdr and how it encodes streams. all config about any streams must be outside of library.
on all streams library makes predictions. be ready to send user controls on bot and then get eeg of user. make library work closer as possible to real cortex.

hardcoded bot gives its sensors and actions together as streams. then from bottom up need to be all streams via their sdr in many levels of hierarchy, like on bottom level lines moving it comes up as pattern of symbols, then on next level bottom symbols moving it comes up as pattern of words, then on next level bottom words moving it comes up as pattern of sentences. and all predictions need to be top down, like sentences words symbols and moving lines on level which then decoded back and intelligent bot takes action on part of stream which it knows action, so intelligent bot sending its sensors and actions as streams, they processed via sdr bottom up in spatiotemporal hierarchy to higher order concepts and then predicted top down next stream, and bot uses actions from this predicted stream.

if not sure how then reread again how sdr works.

you can fully rewrite all this tfjs part to make sdr fully work, use any non standard ways. just make streams of sensors and actions process absolutely similar as in cortex, and to only bots know what they send as actions and what as senses, and make intelligent bot takes predictions from actions to act. make library learn on predictions anomaly as cortex do.

make as possibly full implementation of learning on anomaly and make as many as possible levels of hierarchy. and maybe make first own hierarchy levels for each stream and then combine streams on higher hierarchy level.

make intellectual bot learn not on random actions but from hardcoded bot actions when exploration enabled, but make these exploration rank rewards go to hardcoded bots team when exploration enabled.

show as much as possible data from how intellectual bot mind looks inside, like sdr, patterns, hierarchy, predictions, anomalies, maybe how it changes in time. make ability to autochoose exploration level based on how big anomalies, make higher level anomalies more important as they means that bot don't understands higher concepts.

make maybe audiovisual presentation of bots intentions.

mimicing is good idea, as main idea for these bots to learn players intentions but to better oppose players so it should be some third view mimicing, and idea for this library to copy players intention to put later in artifacts to fulfill most desired intention for artifact holder. but for now idea was only to find how for bot faster learn. so maybe make all bots same and only some will enable hardcoded intentions for learning and other fully hardcoded all time, i mean it should be their hardcoded logic but enabled time to time, and most time when not explore mode they should predict what learned as goal to make them work fully on predictions..

add most important technologies of how sdr works to learn bottom-up spatiotemporal hierarchically on  anomalies in top-down predictions

make deepresearch and add as mush as possible details in implementation of sdr bottom-up spatiotemporal patterns recognition and top-down predictions and anomaly based learning

https://github.com/lucidrains/titans-pytorch/tree/main/titans_pytorch

here attached non-official implementation of titans, self learning transformer.

convert this Sparse Distributed Representations (SDRs) implementation into use transfromers.

convert now to maximum use of gpu and maximum implement what known how brain works, main in learning from streams agnostic to type sensors or actions in bottom-up hierarchy and learn on predictions and anomalies in predictions top-down, so bot will learn intentions having no own intentions initially as cortex works. you can make any changes, my target system in rtx 3060 12GB.

deepresearch, is there any possible use of gpu acceleration to reproduce cortical algorithm.

search how to accelerate on gpu this bot, so it will work maximum similarly to cortex, as has no initial goals and library will not know what are sensors and what are actions and will learn on anomaly of predictions.

convert to no use cpu inside library where streams coming in and predictions goes out, only gpu use in library.

you can make nay changes to library maybe no need htm, just to be anomaly.

make ability to create on maximum grid maximum goals and bots, no need constraints, if such parameters set by user then let it be will be such world somehow, so with any parameters in ui world should start.

make version with full gpu acceleration.

maybe make separate model for every bot, if it will speedup or simplify.

use NeuralMemory, custom gradient handling, advanced memory management techniques like in titans-pytorch.

use cortical concept that only streams of sensors and actions on input and output, so prediction made on both sensors and actions agnostically just as streams, and only bot outside of library decided what are actions. and anomaly in predictions calculated for all these streams both actions and sensors agnostically and per hierarchy level, and most important to fix anomalies in higher hierarchy levels where comes together all streams.

make server.py, index.html, setup and run sh and readme for github but don't place ``` inside blocks, as it ruins aistudio chat markdown.

there no data to learn from backpropagation, as requires realtime learning from very small data, so pytorch-titans must have can solve learning from small data as cortex do.

https://arxiv.org/pdf/2501.00663v1

>> Memory Architectures (MAC/MAG/MAL): These show different ways to integrate the memory module with a standard Transformer/attention core. Since we're building a simpler, stream-only library initially, we'll focus on the core memory update mechanism itself, resembling a recurrent update driven by prediction error.

bot must learn on prediction error -- in is main cortext concept, must be implemented

>> Hierarchy: Implementing a true multi-level hierarchy like HTM with stacked NeuralMemory modules is complex and might exceed the scope/stability needed initially. Decision: Start with a single NeuralMemory layer. The "hierarchy" aspect will be implicit in the depth of the MemoryMLP (or chosen memory model) within the NeuralMemory module. The user can tune the depth parameter of the memory model. True stacking can be a future extension.

for hierarchy stack multiple NeuralMemory levels

implement maximum of article and pytorch-titans in server.py learning library. all calculations must be on server.py backend, index.html frontend via socket only interfacing with users.

don't create any py files except server.py. place all what required from pytorch-titans inside server.py, and rewrite it according to google titans article.

also make ability for players to join simulation operating its own bots but using same actions as used by other bots, so learning bot could learn also from players, so will be mmorpg.

and pressing space scrolling screen. and maybe make play by mouse.

make player bots operates somehow by touchscreen on mobile, as there no keyboard and mouse.

when player join start with fullscreen with touchscreen, mouse, keys controls.

don't place end line comments when concatenating code in javascript in one line as comments applied to code in line and it cause error.

don't concatenate lines in python using ';' as python is not javascript.

make player be able to join any time if there are free learning bots, and when player leaves learning bot returns control back to his setup. so players joining on free learning bots, maybe make when they not joined make them choose which learning bot to join or use any.

make able to just one time click or tap and player bot will continue to move or claim or attack in this direction, so no need to click on each action again.

was error: on next level player bot lost control and disabled button to join. also make ability to leave bot and join any time by any player. when screen reload also player bot lost control, make player session be stored like maybe in frontend settings so will work page reload without player bot lost control.

after some time hardcoded bot stuck near single cell wall for many steps, but then released, and when it stuck moving near learning bot also stuck but after attack from hardcoded bot. it solved after restart. all this with no errors on backend.

was error: when player lost control after new level started. and no ability for player to join any time when there free learning bots. but working ability to leave, and then no ability to join, until simulation stopped.

make not persistent action, as there no ability to easy aim target, but make for controlled bot to follow where tapped and dragged or to where mouse clicked and mouse move, so every simulation step try again to move towards player tap or clicked mouse position location, or activate attack or claiming goal in that direction.

hardcoded bot often stuck near screen border or near one cell wall where no target after this wall so it is not some goal targeting, and then moving near learning bot also stuck but with interaction via attacks, but then learning bot may release from this interaction, but hardcoded bot not unstuck, but one time i seen it unstuck too. after new level hardcoded bot unstuck.

make ability for not joined player to click or tap on free learning bot to take control.
```
