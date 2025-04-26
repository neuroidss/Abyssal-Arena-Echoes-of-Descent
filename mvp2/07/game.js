// game.js

// --- Game Setup ---
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const infoDiv = document.getElementById('info');
const toggleBotModeButton = document.getElementById('toggleBotMode');
const spawnBotManualButton = document.getElementById('spawnBotManual'); // Debug

let canvasWidth = window.innerWidth;
let canvasHeight = window.innerHeight;
canvas.width = canvasWidth;
canvas.height = canvasHeight;

window.addEventListener('resize', () => {
    canvasWidth = window.innerWidth;
    canvasHeight = window.innerHeight;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
});


// --- Game State ---
const player = {
    x: canvasWidth / 2,
    y: canvasHeight / 2,
    radius: 15,
    color: 'blue',
    speed: 3,
    interacting: false,
    controlledByBot: false
};

const target = {
    x: Math.random() * canvasWidth,
    y: Math.random() * canvasHeight,
    radius: 10,
    color: 'green',
    vx: (Math.random() - 0.5) * 2,
    vy: (Math.random() - 0.5) * 2,
    speed: 1
};

const bots = []; // Array to hold bot objects { x, y, radius, color, hierarchy, anomalies }
let botCounter = 0;


// --- Input State ---
const input = {
    up: false,
    down: false,
    left: false,
    right: false,
    interact: false, // Keyboard/Mouse/Touch unified action
    touchMoveX: 0, // From virtual joystick (-1 to 1)
    touchMoveY: 0, // From virtual joystick (-1 to 1)
};

// --- Library Setup ---
const STREAM_NAMES = [
    'playerX', 'playerY',        // Player position (Sense)
    'targetX', 'targetY',        // Target position (Sense)
    'actionMoveX',               // Player intended horizontal move (Action)
    'actionMoveY',               // Player intended vertical move (Action)
    'actionInteract'             // Player interaction attempt (Action)
    // Add more streams here (e.g., 'distanceToTarget', 'targetVelocityX', 'enemyDetected', 'actionShoot')
];

const hierarchyConfig = {
    numLevels: 3,
    learningRate: 0.005, // Smaller learning rate often more stable
    streamNames: STREAM_NAMES,
    layerLatentSizes: [STREAM_NAMES.length, 12, 6], // Example: Input -> 12 -> 6
    anomalyThreshold: 0.08, // Lower threshold for spawning
    botSpawnPatience: 300, // Requires 300 ticks (~5 seconds) of low anomaly
    anomalyHistorySize: 150
};

const playerHierarchy = new CorticalHierarchy(hierarchyConfig);
playerHierarchy.defineStreams(STREAM_NAMES); // Redefine based on constant
playerHierarchy.initializeModels().then(() => {
    console.log("Player Hierarchy Initialized.");
    gameLoop(); // Start game loop only after models are ready
}).catch(err => console.error("Initialization failed:", err));


// --- Data Logging (Optional with Danfo.js) ---
// let dataHistory = []; // Array to store tick data
// const LOG_INTERVAL = 100; // Log every 100 ticks
// let tickCounter = 0;


// --- Input Handling ---

// Keyboard
window.addEventListener('keydown', (e) => {
    switch (e.key.toLowerCase()) {
        case 'w': case 'arrowup':    input.up = true; break;
        case 's': case 'arrowdown':  input.down = true; break;
        case 'a': case 'arrowleft':  input.left = true; break;
        case 'd': case 'arrowright': input.right = true; break;
        case ' ': case 'enter':      input.interact = true; break; // Space or Enter for interaction
    }
});
window.addEventListener('keyup', (e) => {
     switch (e.key.toLowerCase()) {
        case 'w': case 'arrowup':    input.up = false; break;
        case 's': case 'arrowdown':  input.down = false; break;
        case 'a': case 'arrowleft':  input.left = false; break;
        case 'd': case 'arrowright': input.right = false; break;
        case ' ': case 'enter':      input.interact = false; break;
    }
});

// Mouse (Example: Treat click as interact)
canvas.addEventListener('mousedown', () => { input.interact = true; });
canvas.addEventListener('mouseup', () => { input.interact = false; });

// Touch Controls
const joystickArea = document.getElementById('joystick-area');
const joystickKnob = document.getElementById('joystick-knob');
const actionButton = document.getElementById('action-button');
let joystickActive = false;
let joystickStartX = 0;
let joystickStartY = 0;
const joystickRadius = joystickArea.offsetWidth / 2;
const knobRadius = joystickKnob.offsetWidth / 2;

joystickArea.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (e.touches.length === 1) {
        joystickActive = true;
        const touch = e.touches[0];
        const rect = joystickArea.getBoundingClientRect();
        joystickStartX = rect.left + joystickRadius;
        joystickStartY = rect.top + joystickRadius;
         // Calculate initial position relative to center for immediate feedback
         updateJoystick(touch.clientX, touch.clientY);
    }
}, { passive: false });

joystickArea.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (joystickActive && e.touches.length === 1) {
        const touch = e.touches[0];
         updateJoystick(touch.clientX, touch.clientY);
    }
}, { passive: false });

joystickArea.addEventListener('touchend', (e) => {
    e.preventDefault();
     if (joystickActive) {
         joystickActive = false;
         input.touchMoveX = 0;
         input.touchMoveY = 0;
         // Reset knob position
         joystickKnob.style.left = `${joystickRadius - knobRadius}px`;
         joystickKnob.style.top = `${joystickRadius - knobRadius}px`;
     }
}, { passive: false });

function updateJoystick(clientX, clientY) {
    let dx = clientX - joystickStartX;
    let dy = clientY - joystickStartY;
    let distance = Math.sqrt(dx * dx + dy * dy);

    let clampedX = dx;
    let clampedY = dy;

    if (distance > joystickRadius) {
        // Clamp to edge
        clampedX = (dx / distance) * joystickRadius;
        clampedY = (dy / distance) * joystickRadius;
        distance = joystickRadius; // Clamp distance for normalization
    }

     // Normalize to -1 to 1 range
     input.touchMoveX = clampedX / joystickRadius;
     input.touchMoveY = clampedY / joystickRadius; // Y is often inverted in screen coords vs game coords

     // Update knob visual position (relative to joystickArea)
     const knobX = (joystickRadius - knobRadius) + clampedX;
     const knobY = (joystickRadius - knobRadius) + clampedY;
     joystickKnob.style.left = `${knobX}px`;
     joystickKnob.style.top = `${knobY}px`;
}


actionButton.addEventListener('touchstart', (e) => { e.preventDefault(); input.interact = true; actionButton.style.background = 'rgba(200, 0, 0, 0.8)';}, { passive: false });
actionButton.addEventListener('touchend', (e) => { e.preventDefault(); input.interact = false; actionButton.style.background = 'rgba(200, 0, 0, 0.4)';}, { passive: false });


// --- Helper Functions ---
function normalize(value, min, max) {
    if (max === min) return 0; // Avoid division by zero
    return (value - min) / (max - min); // Basic min-max normalization to 0-1
    // Consider other normalization like mapping to -1 to 1 (2 * (...) - 1) if using tanh activations
}

function denormalize(value, min, max) {
     if (max === min) return min;
    return value * (max - min) + min;
}

function normalizeState(state) {
    const normState = {};
    normState.playerX = normalize(state.playerX, 0, canvasWidth);
    normState.playerY = normalize(state.playerY, 0, canvasHeight);
    normState.targetX = normalize(state.targetX, 0, canvasWidth);
    normState.targetY = normalize(state.targetY, 0, canvasHeight);
    normState.actionMoveX = state.actionMoveX; // Already -1 to 1 (or 0)
    normState.actionMoveY = state.actionMoveY; // Already -1 to 1 (or 0)
    normState.actionInteract = state.actionInteract ? 1.0 : 0.0; // Convert boolean to 0/1
    return normState;
}

function drawCircle(x, y, radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.closePath();
}

function updateTarget() {
    target.x += target.vx * target.speed;
    target.y += target.vy * target.speed;

    if (target.x < target.radius || target.x > canvasWidth - target.radius) target.vx *= -1;
    if (target.y < target.radius || target.y > canvasHeight - target.radius) target.vy *= -1;

    target.x = Math.max(target.radius, Math.min(canvasWidth - target.radius, target.x));
    target.y = Math.max(target.radius, Math.min(canvasHeight - target.radius, target.y));
}

async function spawnBot() {
    if (!playerHierarchy.readyToSpawnBot) return; // Ensure condition is met

    const botHierarchy = await playerHierarchy.cloneForBot(); // cloneForBot now returns a promise

    if (botHierarchy) {
        botCounter++;
        const newBot = {
            id: botCounter,
            x: Math.random() * canvasWidth, // Spawn at random position
            y: Math.random() * canvasHeight,
            radius: player.radius * 0.8,
            color: `hsl(${botCounter * 60}, 70%, 50%)`, // Different color for each bot
            hierarchy: botHierarchy,
            anomalies: Array(botHierarchy.numLevels).fill(0),
            lastAction: { actionMoveX: 0, actionMoveY: 0, actionInteract: 0 }, // Store last action taken
            speed: player.speed * 0.8 // Slightly slower
        };
        bots.push(newBot);
        console.log(`Bot ${newBot.id} spawned.`);
    } else {
         console.error("Failed to clone hierarchy for bot.");
    }
}

toggleBotModeButton.addEventListener('click', () => {
    player.controlledByBot = !player.controlledByBot;
    toggleBotModeButton.textContent = player.controlledByBot ? "Switch to Player Control" : "Switch to Bot Control";
    console.log(`Player control switched to: ${player.controlledByBot ? 'Bot' : 'Manual'}`);
});

// Debug: Manual Spawn
spawnBotManualButton.addEventListener('click', async () => {
     console.log("Attempting manual bot spawn...");
     // Temporarily set the flag to allow cloning
     const originalFlag = playerHierarchy.readyToSpawnBot;
     playerHierarchy.readyToSpawnBot = true; // Force allow
     await spawnBot();
     playerHierarchy.readyToSpawnBot = originalFlag; // Restore original flag state
     // We don't reset skill level or counter for manual spawn
});


// --- Game Loop ---
let lastPrediction = {}; // Store last prediction for display
let playerAnomalies = []; // Store last player anomalies

function gameLoop() {
    // --- 1. Clear Canvas ---
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // --- 2. Handle Input & Determine Actions ---
    let moveX = 0;
    let moveY = 0;
    let doInteract = input.interact;

    if (player.controlledByBot) {
        // Use the library's prediction to control the player
        const prediction = lastPrediction; // Use prediction from *previous* tick
        if (prediction && prediction.actionMoveX !== undefined) {
             moveX = Math.sign(prediction.actionMoveX) * (Math.abs(prediction.actionMoveX) > 0.3 ? 1 : 0); // Thresholding prediction
             moveY = Math.sign(prediction.actionMoveY) * (Math.abs(prediction.actionMoveY) > 0.3 ? 1 : 0);
             doInteract = prediction.actionInteract > 0.5; // Thresholding prediction
             // Note: This uses the prediction of *what action the player would take*
        } else {
             // Default to no action if prediction is not ready
             moveX = 0;
             moveY = 0;
             doInteract = false;
        }
    } else {
        // Manual control (Keyboard/Touch)
        moveX = (input.right ? 1 : 0) - (input.left ? 1 : 0);
        moveY = (input.down ? 1 : 0) - (input.up ? 1 : 0);

        // Use touch input if active
        if (Math.abs(input.touchMoveX) > 0.1 || Math.abs(input.touchMoveY) > 0.1) {
             moveX = input.touchMoveX;
             moveY = input.touchMoveY; // Assuming positive Y is down screen, matches game coords
        }
        // Normalize diagonal movement speed
        const mag = Math.sqrt(moveX * moveX + moveY * moveY);
        if (mag > 1) {
            moveX /= mag;
            moveY /= mag;
        }
    }

     // --- 3. Update Player State (Apply Actions & Physics) ---
     const prevPlayerX = player.x;
     const prevPlayerY = player.y;

     player.x += moveX * player.speed;
     player.y += moveY * player.speed;
     player.interacting = doInteract; // This might trigger something visually or in game logic later

     // Boundaries
     player.x = Math.max(player.radius, Math.min(canvasWidth - player.radius, player.x));
     player.y = Math.max(player.radius, Math.min(canvasHeight - player.radius, player.y));

    // --- 4. Update Other Game Elements ---
    updateTarget();

    // --- 5. Prepare Data for Player Hierarchy ---
    const currentRawState = {
        playerX: player.x,
        playerY: player.y,
        targetX: target.x,
        targetY: target.y,
        // ACTIONS taken/intended in this tick:
        actionMoveX: moveX, // The actual command sent (-1 to 1)
        actionMoveY: moveY, // The actual command sent (-1 to 1)
        actionInteract: doInteract ? 1.0 : 0.0
    };
    const currentNormalizedState = normalizeState(currentRawState);


    // --- 6. Process Player Tick ---
    const { anomalies, prediction, rawAnomalies } = playerHierarchy.processTick(currentNormalizedState);
    lastPrediction = prediction; // Store for next frame (if player is bot-controlled) and display
    playerAnomalies = anomalies; // Store average anomalies

    // --- 7. Optional: Log Data ---
    // tickCounter++;
    // if (tickCounter % LOG_INTERVAL === 0 && typeof dfd !== 'undefined') {
    //     dataHistory.push({ ...currentNormalizedState, prediction: prediction, anomalies: anomalies });
    //     try {
    //        let df = new dfd.DataFrame(dataHistory);
    //        console.log("Logged DataFrame:");
    //        df.print();
    //        dataHistory = []; // Clear history after logging
    //     } catch (e) { console.error("Danfo error:", e); }
    // }

    // --- 8. Check for Player Bot Spawning ---
    if (playerHierarchy.shouldSpawnBot()) {
        spawnBot(); // Function now handles the async cloning
    }


    // --- 9. Update Bots ---
    bots.forEach(bot => {
        // a. Get bot's prediction for its *own* action
        const botPrediction = bot.hierarchy.getPrediction(); // Relies on processTick having run for the bot

        let botMoveX = 0;
        let botMoveY = 0;
        let botInteract = false;

        // Use the last stored action if prediction is empty initially
        let actionSource = Object.keys(botPrediction).length > 0 ? botPrediction : bot.lastAction;

         if (actionSource && actionSource.actionMoveX !== undefined) {
             botMoveX = Math.sign(actionSource.actionMoveX) * (Math.abs(actionSource.actionMoveX) > 0.2 ? 1 : 0); // Bot thresholding
             botMoveY = Math.sign(actionSource.actionMoveY) * (Math.abs(actionSource.actionMoveY) > 0.2 ? 1 : 0);
             botInteract = actionSource.actionInteract > 0.5;
         }

        // Normalize diagonal speed
        const botMag = Math.sqrt(botMoveX * botMoveX + botMoveY * botMoveY);
        if (botMag > 1) {
            botMoveX /= botMag;
            botMoveY /= botMag;
        }

        // b. Update bot state (Apply predicted action & Physics)
        bot.x += botMoveX * bot.speed;
        bot.y += botMoveY * bot.speed;
        bot.interacting = botInteract; // Store bot's action state

        // Bot Boundaries
        bot.x = Math.max(bot.radius, Math.min(canvasWidth - bot.radius, bot.x));
        bot.y = Math.max(bot.radius, Math.min(canvasHeight - bot.radius, bot.y));

        // c. Prepare data for bot's *next* tick (current senses + action it just took)
         const botCurrentRawState = {
             playerX: bot.x, // Bot's own position
             playerY: bot.y,
             targetX: target.x, // Bot senses the same target
             targetY: target.y,
             actionMoveX: botMoveX, // The action the bot *just attempted*
             actionMoveY: botMoveY,
             actionInteract: botInteract ? 1.0 : 0.0
         };
         const botCurrentNormalizedState = normalizeState(botCurrentRawState);

        // d. Process bot's tick using its own hierarchy
        const botResult = bot.hierarchy.processTick(botCurrentNormalizedState);
        bot.anomalies = botResult.anomalies; // Update bot's anomalies for display
        bot.lastAction = { // Store the action *taken* by the bot for the next prediction cycle if needed
             actionMoveX: botMoveX,
             actionMoveY: botMoveY,
             actionInteract: botInteract ? 1.0 : 0.0
        };
        // Update bot's prediction *after* processing its current state
        // The prediction for the *next* action is now available via bot.hierarchy.getPrediction() internally,
        // but we capture it at the start of the *next* loop iteration.


    });


    // --- 10. Render ---
    // Player
    drawCircle(player.x, player.y, player.radius, player.color);
    if (player.interacting) { // Visual indicator for interaction
        ctx.strokeStyle = 'yellow';
        ctx.lineWidth = 3;
        ctx.strokeRect(player.x - player.radius - 5, player.y - player.radius - 5, player.radius * 2 + 10, player.radius * 2 + 10);
    }

    // Target
    drawCircle(target.x, target.y, target.radius, target.color);

     // Predicted Player Position (denormalize from prediction)
     if (lastPrediction && lastPrediction.playerX !== undefined) {
         const predX = denormalize(lastPrediction.playerX, 0, canvasWidth);
         const predY = denormalize(lastPrediction.playerY, 0, canvasHeight);
         ctx.beginPath();
         ctx.arc(predX, predY, player.radius, 0, Math.PI * 2);
         ctx.strokeStyle = 'rgba(0, 0, 255, 0.3)'; // Faint blue stroke
         ctx.lineWidth = 2;
         ctx.stroke();
         ctx.closePath();
     }


    // Bots
    bots.forEach(bot => {
        drawCircle(bot.x, bot.y, bot.radius, bot.color);
         if (bot.interacting) { // Bot interaction indicator
             ctx.strokeStyle = 'orange';
             ctx.lineWidth = 2;
             ctx.strokeRect(bot.x - bot.radius - 4, bot.y - bot.radius - 4, bot.radius * 2 + 8, bot.radius * 2 + 8);
         }
    });

    // --- 11. Display Info ---
    let infoText = `Player Skill Level: ${playerHierarchy.skillLevel}<br>`;
    infoText += `Control: ${player.controlledByBot ? 'Bot' : 'Manual'}<br>`;
    infoText += `Bots: ${bots.length}<br>`;
    infoText += `Player Avg Anomaly: ${playerAnomalies.map((a, i) => `L${i}: ${a.toFixed(4)}`).join(' | ')}<br>`;
    infoText += `Spawn Counter: ${playerHierarchy.botSpawnCounter}/${playerHierarchy.botSpawnPatience}<br>`;
    infoText += `Ready To Spawn: ${playerHierarchy.readyToSpawnBot}<br>`;

    bots.forEach(bot => {
        infoText += `<span style="color:${bot.color};">Bot ${bot.id} Anomaly:</span> ${bot.anomalies.map((a, i) => `L${i}: ${a.toFixed(4)}`).join(' | ')}<br>`;
    });

    // Display prediction details (optional)
    // infoText += `<hr>Prediction:<br><pre>${JSON.stringify(lastPrediction, (k,v) => v && v.toFixed ? Number(v.toFixed(3)) : v, 2)}</pre>`;


    infoDiv.innerHTML = infoText;


    // --- 12. Request Next Frame ---
    requestAnimationFrame(gameLoop);
}

// Initial call is now inside playerHierarchy.initializeModels().then(...)
// gameLoop();
