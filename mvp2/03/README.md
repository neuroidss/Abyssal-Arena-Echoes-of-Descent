# Cortical Hierarchy Inspired Learning Library & Demo

This project implements a JavaScript library (`corticalHierarchy.js`) inspired by the concept of spatiotemporal cortical hierarchies. It provides a framework for agents (players or bots) in dynamic environments (like games) to learn patterns, make predictions, and adapt their behavior based on anomaly detection. A simple 2D game demo (`index.html`, `main.js`) showcases its usage.

## Core Concepts

1.  **Unified Action/Sense Stream:** Unlike traditional AI where actions and sensory inputs are often treated separately, this library takes a unified approach. All inputs, whether they originate from user controls (keyboard, mouse, gamepad) or from the game environment's state (position, health, nearby objects), are combined into a single input vector.
2.  **Hierarchical Processing:** The input vector is fed into the lowest level of a neural network hierarchy (modeled using TensorFlow.js Dense layers). Information propagates upwards through the layers.
3.  **Prediction:** The library continuously predicts the *next* state of the unified input vector based on the current state processed through the hierarchy.
4.  **Anomaly Detection:** The core learning signal comes from comparing the prediction with the actual next state that occurs. The difference (e.g., Mean Squared Error) is treated as an "anomaly" score. Lower anomaly means the model is predicting the action/sense stream well.
5.  **Learning:** The anomaly/prediction error is used to train the neural network hierarchy via backpropagation (using `tf.train.adam` and `model.fit`), allowing the agent to improve its predictions over time.
6.  **Skill Level:** A skill metric is calculated based on the recent history of anomaly scores. Consistently low anomaly scores indicate high predictability and thus higher skill.
7.  **Bot Spawning:** When an agent (typically the player) maintains a high skill level (low anomaly) for a sustained period, the library signals that a "bot" can be spawned. This involves cloning the agent's current learned model (network weights) into a new, independent agent instance. This new bot will then act based on the patterns learned by the player.
8.  **Game Agnostic:** The `corticalHierarchy.js` library itself contains no game-specific logic. It only requires a configuration defining the named input streams and their sizes.
9.  **Extensibility:** The structure allows for adding various data streams, potentially including complex sensor data like EEG or EMG in the future, by defining them in the `streamConfig`.

## Library (`corticalHierarchy.js`) API

```javascript
// Constructor
const hierarchy = new CorticalHierarchy({
    streamConfig: { /* ... stream definitions ... */ },
    hiddenLayers: [/* ... layer sizes ... */],
    learningRate: 0.001,
    skillThreshold: 0.95,
    skillHistoryLength: 100,
    // ... other options
});

// Process the current state S(t) to get prediction P(t+1) and anomaly
// Anomaly is based on comparing S(t) with the *previous* prediction P(t)
const result = await hierarchy.processTick(currentInputData);
// result = { predictionObject, currentAnomaly, skillLevel, spawnBotSignal }

// Train the model using the actual outcome S(t+1) for the prediction P(t+1) made in the last processTick
// Needs the input S(t) that *led* to P(t+1) and the target S(t+1)
// The library manages storing the necessary previous input internally now.
const loss = await hierarchy.trainTick(actualNextInputData);

// Get current skill level (0-1, higher is better)
const skill = hierarchy.getSkillLevel();

// Get latest anomaly score object
const anomalies = hierarchy.getAnomalies(); // { overall: value } in V1

// Get the last predicted state as an object
const prediction = hierarchy.getPredictionObject();

// Get/Set model weights (Array of tf.Tensor)
const weights = hierarchy.getWeights();
newBotHierarchy.setWeights(weights);
// Remember to dispose original tensors if needed after cloning/setting

// Dispose TF.js resources when done
hierarchy.dispose();
```

## Demo (`index.html` / `main.js`)

The demo shows:
*   A player controlled by WASD/Mouse (click canvas for pointer lock).
*   The player's actions (movement, looking, firing, interacting) and senses (position, ammo, nearby items/bots) are fed into their `CorticalHierarchy` instance.
*   The hierarchy predicts the next combined action/sense state.
*   Anomaly and skill level are displayed.
*   When the player's skill crosses a threshold, a new bot is spawned using a *clone* of the player's learned model weights.
*   Bots use their own hierarchy's predictions to navigate, sense, and act (shoot, collect ammo).
*   Bots also learn and adapt based on their own prediction errors.
*   A basic "Switch to Bot View" button allows observing the game without controlling the player (and highlights the first bot).

## Setup & Running

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Serve the files:** You need a local HTTP server to run the example due to browser security restrictions (loading JS modules, potential pointer lock).
    *   If you have Python 3:
        ```bash
        python -m http.server
        ```
    *   If you have Node.js and `http-server` installed (`npm install -g http-server`):
        ```bash
        http-server .
        ```
3.  **Open your browser:** Navigate to `http://localhost:8000` (or the port specified by your server).

## Input Streams in Demo

The demo uses the following streams, normalized to `[0, 1]`:

*   `moveX`, `moveY`: Joystick/WASD movement (-1 to 1 mapped to 0 to 1).
*   `lookX`, `lookY`: Mouse/Joystick look (-1 to 1 mapped to 0 to 1).
*   `jump`, `attack`, `interact`, `run`: Button presses (0 or 1).
*   `posX`, `posY`: Player/Bot position normalized by world dimensions.
*   `ammo`, `health`: Player/Bot stats normalized.
*   `nearestEnemyDist`, `nearestEnemyAngle`: Normalized distance and relative angle (0-1) to the closest enemy.
*   `nearestAmmoDist`, `nearestAmmoAngle`: Normalized distance and relative angle (0-1) to the closest ammo pack.

## Limitations & Future Work

*   **Simple Hierarchy:** Uses basic Dense layers. Recurrent layers (LSTM, GRU) might capture temporal dependencies better but add complexity.
*   **Single Anomaly Score:** V1 uses the overall prediction error as the main anomaly. A true hierarchical system would have anomaly scores at each level.
*   **Fixed Input Streams:** The library requires defining all streams at initialization. Dynamically adding/removing streams during runtime would require model restructuring and retraining strategies.
*   **Basic Bot Behavior:** Bots act directly on predictions. More sophisticated behavior could involve planning or goal-setting based on predictions.
*   **Performance:** Training happens in the main thread (though asynchronously). For very complex models or high-frequency updates, Web Workers might be necessary. TF.js memory management relies heavily on `tf.tidy` and careful tensor disposal.
*   **No Explicit Motor/Sensory Distinction:** The library treats all streams equally internally, as requested. The *interpretation* of predicted streams as "actions" (for bots) or "expected senses" happens outside the library.
*   **Hierarchy Growth:** The number of layers is fixed. Dynamically adjusting the hierarchy depth based on learning progress is an advanced topic.

This project provides a foundational framework. Integrating more complex game mechanics, diverse sensor inputs (like EEG), and more sophisticated hierarchical models are potential next steps.