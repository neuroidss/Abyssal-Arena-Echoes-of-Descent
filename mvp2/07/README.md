# Spatiotemporal Cortical Hierarchy Library Demo

This project demonstrates a JavaScript library (`corticalHierarchy.js`) that implements a simplified model of a spatiotemporal cortical hierarchy for processing sensory input and action output, calculating predictions, and detecting anomalies. It includes a playable demo game (`index.html`, `game.js`) showcasing its capabilities, including automatic bot spawning based on learned player behavior.

The library uses TensorFlow.js (`tfjs`) for neural network operations and can optionally integrate with Danfo.js for data logging and analysis (though Danfo.js is not used in the core real-time loop).

## Core Concepts

1.  **Unified Sense/Action Streams:** The library treats all inputs (e.g., player position, target location, keyboard presses) as dimensions in a single input vector. There's no internal distinction between "senses" and "actions".
2.  **Hierarchical Processing:** Data flows up through multiple layers (encoding), and predictions flow down. Each layer attempts to predict its input based on signals from the layer above.
3.  **Prediction & Anomaly:** The core function is prediction. The library predicts the entire next input state (all streams). The difference (e.g., Mean Squared Error) between the prediction and the actual next state is calculated at each hierarchy level, representing the "anomaly" or "prediction error".
4.  **Learning:** The internal models (simple feedforward networks per layer in this version) are continuously trained asynchronously to minimize prediction error (anomaly).
5.  **Bot Spawning (Skill Acquisition):** When the prediction anomalies remain consistently low across all levels for a defined period (indicating the hierarchy accurately models the player's interaction patterns within the environment), the system flags that a "skill" has been learned. It can then "spawn" a bot by cloning the player's current trained hierarchy state. This bot will then use its own hierarchy's predictions to act in the game world.
6.  **Game Agnostic:** The library itself contains no game-specific logic. It only processes named data streams. The interpretation and application of these streams happen in the game code.

## Demo Game

The `index.html` file runs a simple 2D game:

*   **Player:** A blue circle controlled by the user (or the hierarchy's predictions).
*   **Target:** A green circle moving randomly.
*   **Bots:** Circles in other colors, spawned automatically when the player's hierarchy demonstrates stable, low-anomaly predictions. Bots are controlled by their own cloned hierarchy.
*   **Prediction Visualization:** A faint blue outline shows where the player hierarchy *predicts* the player will be in the next frame (based on its prediction of all streams, including player position).
*   **Debug Info:** Displays player skill level, number of bots, anomaly scores per level for the player and each bot, and the bot spawn readiness counter.

### Controls

*   **Keyboard:**
    *   `W/A/S/D` or `Arrow Keys`: Move the player.
    *   `Spacebar` or `Enter`: Trigger the "interact" action (visualized by a yellow square).
*   **Mouse:**
    *   `Click`: Trigger the "interact" action.
*   **Touchscreen:**
    *   **Virtual Joystick (Bottom Left):** Drag to move the player.
    *   **Action Button (Bottom Right):** Tap to trigger the "interact" action.
*   **Buttons:**
    *   `Toggle Player Bot Control`: Switches player control between manual input and the player hierarchy's own predictions. Useful for debugging and seeing how well the hierarchy has learned.
    *   `Spawn Bot Manually`: Clones the current player hierarchy state into a new bot immediately (for testing).

### How it Showcases the Library

*   **Real-time Processing:** The game loop continuously feeds the current game state (player pos, target pos, player actions) into the `playerHierarchy.processTick()` method.
*   **Anomaly Display:** You can see how the anomaly scores change based on player actions and the environment's predictability. When you perform consistent actions (like tracking the target), anomalies should decrease. Unexpected events might increase them.
*   **Prediction:** The faint blue circle shows the library's prediction of the player's next state.
*   **Bot Spawning:** Play consistently for a while (e.g., keep moving, maybe try to follow the target). When the anomaly scores stay low enough for long enough (see `botSpawnPatience`), a new bot will appear, driven by a copy of the hierarchy you just trained through playing.
*   **Bot Behavior:** Observe how the spawned bots behave based on the learned patterns. Their actions are derived from their hierarchy's predictions. Each bot also calculates its own anomalies based on its actions and the environment.

## Library Usage (`corticalHierarchy.js`)

```javascript
// 1. Include tfjs and the library script
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
// <script src="corticalHierarchy.js"></script>

// 2. Define stream names and configuration
const STREAM_NAMES = ['sensor1', 'sensor2', 'action1', 'action2'];
const config = {
    numLevels: 3,
    learningRate: 0.01,
    streamNames: STREAM_NAMES,
    // Optional: define layer sizes explicitly
    // layerLatentSizes: [STREAM_NAMES.length, 10, 5],
    anomalyThreshold: 0.1,
    botSpawnPatience: 200
};

// 3. Create and initialize the hierarchy
const hierarchy = new CorticalHierarchy(config);
hierarchy.defineStreams(STREAM_NAMES); // Necessary if not in config or needs update
await hierarchy.initializeModels(); // IMPORTANT: asynchronous

// 4. In your game loop or update function:
function update() {
    // a. Get current state as an object (use defined stream names)
    //    IMPORTANT: Normalize values (e.g., to 0-1 or -1 to 1)
    const currentState = {
        sensor1: normalize(getSensor1Value(), 0, 100),
        sensor2: normalize(getSensor2Value(), -50, 50),
        action1: getAction1Value(), // e.g., -1, 0, or 1
        action2: isAction2Active() ? 1.0 : 0.0
    };

    // b. Process the tick
    const { anomalies, prediction, rawAnomalies } = hierarchy.processTick(currentState);
    // 'anomalies' contains the average anomaly score per level
    // 'prediction' is an object with predicted values for the next state's streams (denormalize if needed)
    // 'rawAnomalies' contains the anomaly for the current tick only

    // c. Use anomalies/prediction (e.g., display, control AI)
    console.log("Avg Anomalies:", anomalies);
    // let predictedSensor1 = denormalize(prediction.sensor1, 0, 100);

    // d. Check for bot spawning condition
    if (hierarchy.shouldSpawnBot()) {
        const botHierarchy = await hierarchy.cloneForBot();
        if (botHierarchy) {
            // Create a game bot controlled by botHierarchy
            // Remember the clone resets the player's spawn counter/flag
        }
    }

    requestAnimationFrame(update);
}

// Start the loop after initialization
// update();
```

## Setup & Running
Dependencies: The demo only requires a modern web browser that supports JavaScript and Canvas. TensorFlow.js is loaded via CDN.

Serve Files: Because the game loads JavaScript modules (game.js, corticalHierarchy.js), you need to serve the files using a local web server. Browsers often restrict loading modules directly from the file:// protocol.

Using Python:
```bash
# Navigate to the project directory in your terminal
cd /path/to/your/project/folder
# Start Python's built-in HTTP server (Python 3)
python -m http.server
# Or for Python 2
# python -m SimpleHTTPServer
```

Using Node.js: Install http-server globally (npm install -g http-server) and run:
```bash
# Navigate to the project directory
cd /path/to/your/project/folder
# Start the server
http-server
```

Open in Browser: Open your web browser and navigate to http://localhost:8000 (or the port specified by your server, usually 8000 or 8080). You should see the game running.

## Future Extensions / Ideas
Dynamic Stream Handling: Allow adding/removing streams during runtime (requires more complex model resizing/adaptation).

Dynamic Hierarchy Depth: Automatically add more levels if learning plateaus.

More Sophisticated Models: Use recurrent layers (LSTM, GRU) within CorticalLayer for better temporal pattern learning.

Attention Mechanisms: Incorporate attention to focus processing on relevant streams.

EEG/BCI Integration: Add streams for EEG data features.

Persistence: Save/load trained hierarchy weights (getWeights, setWeights).

Refined Bot Behavior: Implement more complex goal-seeking behavior for bots using the hierarchy's predictions and anomalies.

External Data Analysis: Use the logged Danfo.js DataFrames for offline analysis of learning progress and anomalies.

# vibe coding prompt
```
first you can search spatiotemporal cortical hierarchy.

make game library where tfjs and danfojs, which collecting what user sense and how user act. user data can be mobile and desktop controls.
make single html file game example where to use this library.
first make plan then make code.
make library able to connect any number of actions and senses from user and from game state which user senses. what player try to do are his action, but how his action appear in game are senses.
make temporary spatial hierarchical levels of sensory cortex and motor cortex, because player may try to move through wall l it will be action, but he will sense that there no movement because wall not movable.
collect dataframe of actions and senses.
process dataframe via temporary spatial hierarchical levels.
first level is encoding all senses and actions.
so on upper levels will be formed patterns.
make predictions of what will be inputs on lower levels based on outputs of higher levels, so from outputs from highest level by this backward way will be formed prediction of next dataframe of user actions and senses which are on lowest level inputs.
form anomaly scores between predicted and actual state on every cortical level.

predictions on dataframe will become next possible actions. and if senses and actions okay with predictions then no anomaly, but if anomaly then relearn new patterns.
after no anomalies long time spawn bot forked from what neural network learned on how player actions lead to which changes in senses, so it will act as player but as separate bot. so for bot will be also its actions and senses in dataframe and predictions with anomalies, how bot is good in predicting and how low anomalies is bot level, higher level anomalies more important in calculating of bot skill level, it may be how much sigma, level of significance.
also make player debug ability to switch to bot mode.
training and finetunnig should be asynchronously automatically processed when needed.
every anomaly calculated when new action and senses ready.
make same anomaly score for player as for bot. and spawn bot when skill level increased by one on skill scale restart this skill level increase calculation every time when skill level falls, so prediction goes to next level.

make some playable game demo, to show actions how to interact and senses what in world can be changed any time outside of library and library should adapt. there will be llm tool tool_creation_tool (https://github.com/neuroidss/ToolArtifact) which will make all other llm tools and will be tool to create artifacts from any tool and these artifacts will be used by players and bot to sense and act in mmorpg. and then will be added brain computer interface on base of 32 channel EEG and ingame llm will search science articles about working memory in brain and will feed eeg data to this new anomaly predictions library, so eeg will be next possible sensory and actions data, and bots will have artifacts to sense players intentions directly from players eeg, and players will adapt to this super predictions, as even players themself don't know their intentions fully but bots will know from players eeg.

make easy to add like eeg or emg into senses or electrical stimulation of brain or muscles or vibromotor sensory substitution to actions.
main to make fully functional library with ability to predict and ability to make trained model clone to drive bots. all other game example is not important but just show what will be in full game using this library.

not include any game specific inside library. library should be universal framework, fully implement most important features of library for most flexibility and dynamic changes.

must be single js file with library. make example of use where to show most important features which can't be done without this library. make readme for github with spaces before ``` blocks to not break chat markdown. make sh files how to setup and run example.

in library should be no difference between actions and senses as all them creating anomaly detection, and actions are just what used to move bots. make in game example ability to play with touchscreen mobile ready, keyboard, mouse.

make absolutely no difference between senses and actions inside library, as cortex mot make this difference. and cortex makes predictions on both of them, and only outside of cortex there periphery decides what only sensing and what only acting. actions are just predictions for cortex, senses restored via predictions to make senses more stable with less information from environment using more stable patterns.

make streams adding and removing available any time, like make for new stream additional proportional part in layers, so make ability to resize, scale layers if necessary. make first layers for each stream more connected with this stream but later all streams interconnected on higher levels. make also ability to change type of layer if possible.

use dataframes only outside of cortex for faster processing.

maybe decide when to add additional levels of hierarchy, is child when learning first having less hierarchy levels than adults. so at start it may be less levels for faster starting to get result, but at some point when not enough effectivity increase then make more cortical levels.

add touchscreen controls for mobile.
```
