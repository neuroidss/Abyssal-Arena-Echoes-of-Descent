 # SpatioTemporal Cortical Hierarchy (STCH) Library Demo

 This project demonstrates a JavaScript library (`stch.js`) inspired by computational neuroscience theories of the cortex, specifically Hierarchical Temporal Memory (HTM) and Predictive Coding. The library aims to learn patterns from sensory inputs and action outputs, predict future states, and detect anomalies. Learned models can be used to drive autonomous agents (bots).

 The demo is a simple 2D top-down shooter game (`index.html`) where the library learns from the player's actions and the resulting game state changes. When the library achieves stable predictions (low anomaly scores), it spawns a bot that attempts to mimic the learned behavior.

 ## Core Concepts

 *   **Streams:** The library accepts multiple named input streams, representing both sensory information (what the entity perceives) and action information (what the entity intends to do). Examples: `playerX`, `playerY`, `nearestEnemyAngle`, `moveX`, `attackButton`.
 *   **Uniform Processing:** Internally, senses and actions are treated as part of a single input vector. The hierarchy learns patterns and dependencies across all streams simultaneously.
 *   **Hierarchy:** The library uses multiple levels of processing, implemented as TensorFlow.js autoencoder models. Lower levels handle more concrete details, while higher levels learn more abstract representations.
 *   **Prediction:** Based on the current state encoded in the hierarchy, the library predicts the *entire* input vector for the next timestep (both senses and actions).
 *   **Anomaly Detection:** The prediction is compared to the actual next input state at each level of the hierarchy. Significant differences generate anomaly scores.
 *   **Learning:** High anomaly scores trigger asynchronous training of the models at the respective levels, allowing the hierarchy to adapt to new patterns or unexpected events.
 *   **Bot Spawning:** When the overall anomaly score remains low for a period (indicating accurate predictions and stable learning), the library can "spawn" a bot by cloning its current internal model state.
 *   **Bot Control:** The spawned bot uses its cloned hierarchy to generate actions. It feeds its current sensory input into its hierarchy and uses the *predicted* action components from the output as its next move. Bots also calculate their own anomalies, reflecting how well their internal model matches their current situation.

 ## Files

 *   `stch.js`: The core library implementing `STCHierarchy` and `Bot` classes. Uses TensorFlow.js.
 *   `index.html`: A single-file HTML game demo that includes and uses `stch.js`. Contains game logic, rendering (Canvas API), input handling (Keyboard, Mouse, Gamepad, Touch), and integrates with the STCH library.
 *   `README.md`: This file.

 ## Features Demonstrated in Example

 *   **Unified Input:** Handles keyboard, mouse (pointer lock for look), gamepad, and on-screen touch controls.
 *   **STCH Integration:**
    *   Collects player position, orientation, health, ammo, and relative positions of nearest enemy/ammo pack as **senses**.
    *   Collects player movement inputs, look inputs, and button presses (run, jump, attack, interact) as **actions**.
    *   Feeds combined senses and actions into the `STCHierarchy`.
    *   Displays anomaly scores per level for the player.
 *   **Bot Spawning:** Automatically spawns a bot when the player's hierarchy shows stable predictions.
 *   **Bot Behavior:** Bots use their cloned hierarchy to navigate, seek ammo, and attack the player based on learned patterns. Bot skill (based on anomaly) is displayed.
 *   **Debug Control Switch:** Press 'P' to toggle control between the player and the most recently spawned *alive* bot. This allows observing the bot's behavior directly using player input mechanisms fed through the bot's learned model (though the demo currently uses player input directly on the possessed bot).

 ## How to Run

 1.  **No Build Step Needed:** The demo uses CDNs for TensorFlow.js and includes the library directly.
 2.  **Serve Locally:** Because the Gamepad API and potentially Pointer Lock work more reliably when served over HTTP/S (even locally), you need a simple web server.
    *   If you have Node.js/npm:
        ```bash
        npx http-server .
        ```
    *   If you have Python 3:
        ```bash
        python -m http.server
        ```
    *   Open your browser and navigate to the local address provided (e.g., `http://localhost:8080` or `http://127.0.0.1:8000`).
 3.  **Interact:**
    *   **Keyboard/Mouse:** WASD (move), Mouse (look - click canvas to lock), Shift (run), Space (jump - currently no effect), Left Click (attack), E (interact/pickup), Q (special - no effect), F (burst - no effect).
    *   **Gamepad:** Left Stick (move), Right Stick (look), RB (run), A (jump - no effect), B (attack), X (interact), Y (burst - no effect), RT (special - no effect). *(Note: Gamepad mappings might vary slightly)*
    *   **Touch:** Left virtual joystick (move), Right virtual joystick (look), On-screen buttons (Y, X, B, RB, A, RT - mapping matches gamepad where applicable).
    *   **Debug:** Press 'P' to switch control between player and an available bot.

 ## Library Usage (stch.js API)

 ```javascript
 // 1. Include tfjs and stch.js

 // 2. Initialize TFJS backend (do this once)
 await tf.ready();
 tf.setBackend('webgl'); // or 'wasm', 'cpu'

 // 3. Define Stream Configuration
 const streamConfig = {
     // Senses (value is dimension)
     'playerX': 1, 'playerY': 1, 'playerAngle': 1, 'playerHealth': 1, 'playerAmmo': 1,
     'nearestEnemyDist': 1, 'nearestEnemyAngle': 1, 'nearestAmmoDist': 1, 'nearestAmmoAngle': 1,
     // Actions
     'moveX': 1, 'moveY': 1, 'lookX': 1, 'run': 1, 'jump': 1,
     'attack': 1, 'special': 1, 'interact': 1, 'burst': 1
 };

 // 4. Define Hierarchy Structure and Parameters
 const hierarchyConfig = {
     streams: streamConfig,
     hierarchyParams: [ // Layers: encodingDim, anomalyThreshold
         { encodingDim: 48, anomalyThreshold: 0.08 },
         { encodingDim: 24, anomalyThreshold: 0.04 },
         { encodingDim: 12, anomalyThreshold: 0.02 }
     ],
     learningRate: 0.005,
     stabilityWindow: 150,    // Steps for stability check
     stabilityThreshold: 0.005, // Avg anomaly level for stability
     botSpawnCooldown: 300    // Min steps between spawns
 };

 // 5. Create Hierarchy Instance (pass tf instance)
 let hierarchy = new STCHierarchy(hierarchyConfig, tf);

 // --- In your game loop ---

 // 6. Prepare Input Data (object map, values MUST be arrays)
 const currentInputData = {
    'playerX': [pX], 'playerY': [pY], /* ... all other senses ... */
    'moveX': [mX], 'moveY': [mY], /* ... all other actions ... */
 };

 // 7. Process Step (async)
 let spawnedBotHierarchy = null;
 if (hierarchy && hierarchy.isBuilt) {
    try {
        const results = await hierarchy.processStep(currentInputData);
        const anomalies = results.anomalies; // Array of anomaly scores per level
        const prediction = results.prediction; // Object map of predicted next state streams
        spawnedBotHierarchy = results.botSpawned; // Null or a cloned STCHierarchy instance

        // Use anomalies for feedback, prediction for other purposes if needed
        // console.log("Anomalies:", anomalies);

    } catch (error) {
        console.error("STCH Error:", error);
    }
 }

 // 8. Handle Bot Spawning
 if (spawnedBotHierarchy) {
     const botId = generateUniqueId();
     // Define which streams the bot controls
     const botActionStreams = ['moveX', 'moveY', 'lookX', 'run', 'attack', 'interact'];
     const gameBot = new Bot(botId, spawnedBotHierarchy, botActionStreams);
     // Add gameBot to your game's bot management system
 }

 // --- For Bots ---

 // 9. Get Bot Action (async)
 const botSenses = getBotSenseData(gameBot); // Get senses relevant to the bot
 try {
    const predictedActions = await gameBot.getAction(botSenses);
    // Apply predictedActions (e.g., predictedActions.moveX) to the bot in the game world
    // gameBot.currentAnomalies contains the bot's internal anomalies for this step
    // gameBot.skillLevel contains a metric based on recent anomalies

 } catch (error) {
    console.error("Bot Action Error:", error);
 }

 // 10. Cleanup (Optional but recommended for SPA)
 // hierarchy.dispose();
 // gameBot.dispose(); // Disposes the bot's internal hierarchy too
 ```

 ## Future Possibilities / TODO

 *   More sophisticated anomaly calculation (e.g., sequence likelihood).
 *   Dynamic addition/removal of streams during runtime (complex).
 *   Dynamic hierarchy resizing (adding/removing levels based on learning progress).
 *   More complex bot behaviors emerging from the learned models.
 *   Integration with external data sources like EEG/EMG (would require mapping signals to input streams).
 *   Using the prediction stream more actively (e.g., for anticipating opponent moves).
 *   More robust error handling and state management.
 *   Performance optimization (model quantization, batching if applicable).

