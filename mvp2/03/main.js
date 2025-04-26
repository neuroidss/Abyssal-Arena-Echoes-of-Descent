// Ensure libraries (tf.js, danfo.js, corticalHierarchy.js) are loaded before this

// --- Configuration ---
const config = {
    streamConfig: {
        // Player Actions (Normalized 0-1)
        'moveX': 1,        // Left (-1 -> 0) / Right (1 -> 1) -- Simple mapping for demo
        'moveY': 1,        // Backward (-1 -> 0) / Forward (1 -> 1)
        'lookX': 1,        // Look Left (-1 -> 0) / Right (1 -> 1)
        'lookY': 1,        // Look Down (-1 -> 0) / Up (1 -> 1)
        'jump': 1,         // 0 or 1
        'attack': 1,       // 0 or 1
        'interact': 1,     // 0 or 1
        'run': 1,          // 0 or 1 (sprint)
        // Player Senses (Normalized 0-1)
        'posX': 1,         // Player X position (scaled to 0-1 based on world bounds)
        'posY': 1,         // Player Y position (scaled to 0-1 based on world bounds)
        'ammo': 1,         // Current ammo count (scaled 0-1)
        'health': 1,       // Current health (scaled 0-1)
        // Simplified Enemy/Item Senses (Normalized 0-1)
        'nearestEnemyDist': 1, // Distance to nearest enemy (scaled, 1 if none)
        'nearestEnemyAngle': 1, // Angle to nearest enemy (0-1, 0 if none)
        'nearestAmmoDist': 1, // Distance to nearest ammo (scaled, 1 if none)
        'nearestAmmoAngle': 1, // Angle to nearest ammo (0-1, 0 if none)
    },
    hiddenLayers: [48, 24], // Example hierarchy
    learningRate: 0.002,
    skillThreshold: 0.90, // Lower threshold for easier bot spawning in demo
    skillHistoryLength: 50,
    activation: 'relu',
    finalActivation: 'sigmoid' // Use sigmoid for 0-1 range
};

const WORLD_WIDTH = 800;
const WORLD_HEIGHT = 600;
const PLAYER_SPEED = 2;
const PLAYER_SPRINT_SPEED = 4;
const PLAYER_TURN_RATE = 0.05; // Radians per input unit
const MAX_AMMO = 20;

// --- Game State ---
let player = {
    x: WORLD_WIDTH / 2,
    y: WORLD_HEIGHT / 2,
    angle: 0, // Radians
    ammo: 10,
    health: 1.0,
    vx: 0, // Velocity for smooth movement
    vy: 0,
    va: 0, // Angular velocity
    isRunning: false,
    isAttacking: false,
    isJumping: false, // Simple flag for demo
    isInteracting: false
};

let bots = []; // Array to hold { id, x, y, angle, ammo, health, hierarchy, lastInputData }
let ammoItems = [];
let projectiles = []; // { x, y, angle, ownerId }
let nextBotId = 0;

let playerHierarchy = null;
let gameRunning = false;
let lastTimestamp = 0;
let botViewActive = false; // For debug view switching
let targetBotId = -1;

// Input state (normalized 0-1 where applicable)
let inputs = {
    moveX: 0.5, moveY: 0.5, lookX: 0.5, lookY: 0.5,
    jump: 0, attack: 0, interact: 0, run: 0
};
let keys = {}; // Keyboard state

// Danfo.js DataFrame for logging (optional)
let logData = [];
const logInterval = 60; // Log every 60 frames
let frameCount = 0;

// --- DOM Elements ---
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');
const playerStatusDiv = document.getElementById('playerStatus');
const botStatusDiv = document.getElementById('botStatus');
const toggleBotViewButton = document.getElementById('toggleBotViewButton');

// --- Initialization ---
function init() {
    canvas.width = WORLD_WIDTH;
    canvas.height = WORLD_HEIGHT;

    // Spawn initial ammo
    for (let i = 0; i < 5; i++) {
        spawnAmmo();
    }

    // Setup Input Listeners
    setupInputListeners();

    // Create Player Hierarchy
    playerHierarchy = new CorticalHierarchy(config);

    // Start game loop
    gameRunning = true;
    lastTimestamp = performance.now();
    requestAnimationFrame(gameLoop);

    console.log("Game Initialized. Hierarchy created.");
    statusDiv.textContent = "Status: Initialized. Move with WASD, Look with Mouse (click canvas), Attack=LMB, Jump=Space, Interact=E, Run=Shift";
}

// --- Input Handling ---
function setupInputListeners() {
    // Keyboard
    window.addEventListener('keydown', (e) => keys[e.code] = true);
    window.addEventListener('keyup', (e) => keys[e.code] = false);

    // Mouse Look (Pointer Lock)
    canvas.addEventListener('click', () => {
        if (!document.pointerLockElement) {
            canvas.requestPointerLock().catch(err => console.error("Pointer lock failed:", err));
        }
    });
    document.addEventListener('pointerlockchange', () => {
        if (document.pointerLockElement !== canvas) {
            console.log("Pointer lock lost");
            // Reset look input if needed
            inputs.lookX = 0.5;
            inputs.lookY = 0.5;
        }
    });
    document.addEventListener('mousemove', (e) => {
        if (document.pointerLockElement === canvas && !botViewActive) {
            // Normalize mouse movement - adjust sensitivity as needed
            inputs.lookX = Math.max(0, Math.min(1, 0.5 + e.movementX * 0.01));
            inputs.lookY = Math.max(0, Math.min(1, 0.5 - e.movementY * 0.01)); // Invert Y
        }
    });

    // Mouse Buttons
    canvas.addEventListener('mousedown', (e) => {
        if (document.pointerLockElement === canvas && !botViewActive) {
            if (e.button === 0) inputs.attack = 1; // Left Mouse Button
            // Add other buttons if needed
        }
    });
    canvas.addEventListener('mouseup', (e) => {
         if (document.pointerLockElement === canvas && !botViewActive) {
            if (e.button === 0) inputs.attack = 0;
         }
    });

    // Touch Controls (Placeholder - requires UI elements)
    // TODO: Add listeners for on-screen joysticks/buttons

    // Gamepad API (Simplified)
    // TODO: Poll gamepad state in game loop updateInputs()

    // Button Listener
    toggleBotViewButton.addEventListener('click', () => {
        botViewActive = !botViewActive;
        targetBotId = botViewActive && bots.length > 0 ? bots[0].id : -1; // Target first bot
        toggleBotViewButton.textContent = botViewActive ? "Switch to Player View" : "Switch to Bot View";
        console.log("Bot view active:", botViewActive, "Target:", targetBotId);
    });
}

function updateInputs() {
    // Keyboard WASD for movement
    let targetMoveX = 0.5;
    let targetMoveY = 0.5;
    if (keys['KeyA'] || keys['ArrowLeft']) targetMoveX = 0; // Left
    if (keys['KeyD'] || keys['ArrowRight']) targetMoveX = 1; // Right
    if (keys['KeyW'] || keys['ArrowUp']) targetMoveY = 1; // Forward
    if (keys['KeyS'] || keys['ArrowDown']) targetMoveY = 0; // Backward
    inputs.moveX = inputs.moveX * 0.8 + targetMoveX * 0.2;
    inputs.moveY = inputs.moveY * 0.8 + targetMoveY * 0.2;

    // Keyboard for other actions
    inputs.jump = (keys['Space']) ? 1 : 0;
    inputs.interact = (keys['KeyE']) ? 1 : 0;
    inputs.run = (keys['ShiftLeft'] || keys['ShiftRight']) ? 1 : 0;
    // Attack is handled by mouse down/up

    // Reset look input ONLY if pointer lock is NOT active
    // This allows the mousemove listener to fully control lookX/Y when locked.
    if (document.pointerLockElement !== canvas) {
         inputs.lookX = inputs.lookX * 0.9 + 0.5 * 0.1; // drift back to center slowly
         inputs.lookY = inputs.lookY * 0.9 + 0.5 * 0.1;
    }
    // else {
    //     // Optional: Could add a check here: if NO mouse movement detected
    //     // for X milliseconds while locked, then start drifting. More complex.
    //     // For now, just let the mousemove dictate entirely when locked.
    // }


    // --- Log for debugging mouse rotation ---
    if (document.pointerLockElement === canvas) {
         console.log(`Pointer Locked. inputs.lookX: ${inputs.lookX.toFixed(3)}`);
    }
    // --- End Log ---


    // TODO: Poll Gamepad state and map to inputs object (normalize)
    // const gamepads = navigator.getGamepads();
    // if (gamepads[0]) { ... map axes and buttons ... }
    console.log("Inputs updated:", JSON.stringify(inputs)); // <-- ADD THIS
}

// --- Normalization Helpers ---
function normalize(value, min, max) {
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
}
function denormalize(value, min, max) {
    return value * (max - min) + min;
}
function normalizeAngle(angle) { // Normalize angle to 0-1
    return (angle % (2 * Math.PI) + (2 * Math.PI)) % (2 * Math.PI) / (2 * Math.PI);
}
function denormalizeAngle(normAngle) { // Denormalize 0-1 to angle
     return normAngle * 2 * Math.PI;
}

// --- Sensing Helpers ---
function getSenses(entity) {
    const senses = {};
    // Basic position/state
    senses.posX = normalize(entity.x, 0, WORLD_WIDTH);
    senses.posY = normalize(entity.y, 0, WORLD_HEIGHT);
    senses.ammo = normalize(entity.ammo, 0, MAX_AMMO);
    senses.health = entity.health; // Assuming health is already 0-1

    // Find nearest entities (simple distance check)
    let nearestEnemyDist = Infinity;
    let nearestEnemyAngle = 0;
    let nearestAmmoDist = Infinity;
    let nearestAmmoAngle = 0;

    const checkEntity = (target, type) => {
        const dx = target.x - entity.x;
        const dy = target.y - entity.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < (type === 'enemy' ? nearestEnemyDist : nearestAmmoDist)) {
             if (type === 'enemy') nearestEnemyDist = dist;
             else nearestAmmoDist = dist;

            let angle = Math.atan2(dy, dx) - entity.angle; // Relative angle
             if (type === 'enemy') nearestEnemyAngle = normalizeAngle(angle);
             else nearestAmmoAngle = normalizeAngle(angle);
        }
    };

    // Check against player (if entity is a bot) or bots (if entity is player)
    if (entity === player) {
        bots.forEach(bot => checkEntity(bot, 'enemy'));
    } else {
        checkEntity(player, 'enemy'); // Bot senses the player
        bots.forEach(bot => { if (bot.id !== entity.id) checkEntity(bot, 'enemy')}); // Bots sense each other
    }

    // Check against ammo items
    ammoItems.forEach(item => checkEntity(item, 'ammo'));

    // Normalize distances (e.g., based on max world diagonal)
    const maxDist = Math.sqrt(WORLD_WIDTH * WORLD_WIDTH + WORLD_HEIGHT * WORLD_HEIGHT);
    senses.nearestEnemyDist = nearestEnemyDist === Infinity ? 1.0 : normalize(nearestEnemyDist, 0, maxDist);
    senses.nearestEnemyAngle = nearestEnemyDist === Infinity ? 0.0 : nearestEnemyAngle;
    senses.nearestAmmoDist = nearestAmmoDist === Infinity ? 1.0 : normalize(nearestAmmoDist, 0, maxDist);
    senses.nearestAmmoAngle = nearestAmmoDist === Infinity ? 0.0 : nearestAmmoAngle;

    return senses;
}


// --- Game Objects & Actions ---
function spawnAmmo() {
    ammoItems.push({
        x: Math.random() * WORLD_WIDTH,
        y: Math.random() * WORLD_HEIGHT,
        size: 5
    });
}

function fireProjectile(owner) {
     if (owner.ammo > 0) {
        owner.ammo--;
        projectiles.push({
            x: owner.x + Math.cos(owner.angle) * 15, // Start slightly ahead
            y: owner.y + Math.sin(owner.angle) * 15,
            angle: owner.angle,
            speed: 8,
            ownerId: owner.id, // player might need an id=0 or similar
            life: 50 // Frames to live
        });
        // console.log(`${owner.id === undefined ? 'Player' : 'Bot '+owner.id} fired. Ammo: ${owner.ammo}`);
     }
}

// --- Update Game State ---
function update(dt) {
    const friction = 0.90; // Slow down velocity
    const angularFriction = 0.85;

    // --- Player Update ---
    if (!botViewActive) {
//        updateInputs(); // Get latest keyboard/mouse state
        console.log("Applying player controls. Inputs:", JSON.stringify(inputs)); // <-- ADD 1

        // Apply inputs to player velocity/rotation
        let currentSpeed = inputs.run > 0.5 ? PLAYER_SPRINT_SPEED : PLAYER_SPEED;
        // Denormalize: -1 for 0, 0 for 0.5, +1 for 1
        let moveForward = (inputs.moveY - 0.5) * 2;
        let moveSideways = (inputs.moveX - 0.5) * 2;
        let lookHorizontal = (inputs.lookX - 0.5) * 2;

        // Increase turn sensitivity slightly maybe?
        const turnSensitivityFactor = 1.0; // Adjust this multiplier (e.g., 1.0, 1.2, 0.8)
        let target_va = lookHorizontal * PLAYER_TURN_RATE * turnSensitivityFactor;

        // Calculate target velocity based on input and player angle
        // Using direct vector math might be clearer for strafing+forward
        let moveAngle = player.angle; // Angle to move forward/backward
        let strafeAngle = player.angle + Math.PI / 2; // Angle to move sideways

        let target_vx = (Math.cos(moveAngle) * moveForward + Math.cos(strafeAngle) * moveSideways) * currentSpeed;
        let target_vy = (Math.sin(moveAngle) * moveForward + Math.sin(strafeAngle) * moveSideways) * currentSpeed;

        // Smoothly interpolate towards target velocity/angular velocity
        const accel = 0.15; // Acceleration factor (adjust 0.1-0.5)
        player.vx = player.vx * (1 - accel) + target_vx * accel;
        player.vy = player.vy * (1 - accel) + target_vy * accel;
        player.va = player.va * (1 - accel) + target_va * accel;


        // Apply actions (Attack is handled in input listeners now)
        player.isRunning = inputs.run > 0.5;
        player.isJumping = inputs.jump > 0.5; // Just a flag
        player.isInteracting = inputs.interact > 0.5;
        // Remove attack logic from here if handled in listener directly
        // if (inputs.attack > 0.5 && !player.isAttacking) { // Fire on press
        //      fireProjectile(player);
        //      player.isAttacking = true; // Prevent holding down fire rapid fire
        // } else if (inputs.attack < 0.5) {
        //     player.isAttacking = false;
        // }

    } else {
         // If bot view is active, decay player velocity
         player.vx *= friction;
         player.vy *= friction;
         player.va *= angularFriction;
    }

     // Apply physics to player (velocity changes position)
    player.x += player.vx; // Assuming dt=1 for simplicity now, velocity IS the change per frame
    player.y += player.vy;
    player.angle += player.va;
    // Apply friction (dampening) AFTER applying movement for this frame
    player.vx *= friction;
    player.vy *= friction;
    player.va *= angularFriction;

    // Boundary checks for player
    player.x = Math.max(0, Math.min(WORLD_WIDTH, player.x));
    player.y = Math.max(0, Math.min(WORLD_HEIGHT, player.y));
    player.angle = (player.angle + 2 * Math.PI) % (2 * Math.PI);


    // --- Bot Updates ---
    bots.forEach(bot => {
        // Bot uses its hierarchy's prediction to act
        const prediction = bot.hierarchy.getPredictionObject(); // Get P(t+1)

        if (prediction) {
            // Denormalize predicted actions
            let botMoveForward = denormalize(prediction.moveY, -1, 1);
            let botMoveSideways = denormalize(prediction.moveX, -1, 1);
            let botLook = denormalize(prediction.lookX, -1, 1);
            let botRun = prediction.run > 0.5;
            let botAttack = prediction.attack > 0.5;
            let botInteract = prediction.interact > 0.5;

            let botSpeed = botRun ? PLAYER_SPRINT_SPEED : PLAYER_SPEED;

            // Apply predicted actions (similar physics to player)
             // Bot needs its own velocity/angular velocity if using physics
            // Simplified: Directly set position based on prediction for demo
            const moveAngle = Math.atan2(botMoveForward, botMoveSideways) - Math.PI / 2; // Angle of move intent
            const moveMagnitude = Math.min(1, Math.sqrt(botMoveForward*botMoveForward + botMoveSideways*botMoveSideways)); // Clamp magnitude

            bot.x += Math.cos(bot.angle) * botMoveForward * botSpeed * dt * 0.5; // Simplified movement
            bot.y += Math.sin(bot.angle) * botMoveForward * botSpeed * dt * 0.5;
            // bot.x += Math.cos(bot.angle + moveAngle) * moveMagnitude * botSpeed * dt; // Strafe-style movement
            // bot.y += Math.sin(bot.angle + moveAngle) * moveMagnitude * botSpeed * dt;
            bot.angle += botLook * PLAYER_TURN_RATE * dt;

            bot.angle = (bot.angle + 2 * Math.PI) % (2 * Math.PI);
            bot.x = Math.max(0, Math.min(WORLD_WIDTH, bot.x));
            bot.y = Math.max(0, Math.min(WORLD_HEIGHT, bot.y));

            if (botAttack && !bot.isAttacking) { // Simple check to prevent rapid fire
                fireProjectile(bot);
                bot.isAttacking = true;
            } else if (!botAttack) {
                bot.isAttacking = false;
            }

            // Interaction (bot picks up ammo)
            if (botInteract) {
                 ammoItems.forEach((item, index) => {
                    const dx = item.x - bot.x;
                    const dy = item.y - bot.y;
                    if (dx * dx + dy * dy < (item.size + 10) * (item.size + 10)) { // 10 is bot radius
                        bot.ammo = Math.min(MAX_AMMO, bot.ammo + 5);
                        ammoItems.splice(index, 1);
                        spawnAmmo(); // Respawn one
                        console.log(`Bot ${bot.id} picked up ammo. Ammo: ${bot.ammo}`);
                    }
                });
            }
        }
    });


    // --- Projectile Update & Collision ---
    projectiles = projectiles.filter(p => {
        p.x += Math.cos(p.angle) * p.speed * dt;
        p.y += Math.sin(p.angle) * p.speed * dt;
        p.life--;

        // Check collision with player
        if (p.ownerId !== undefined) { // Check if projectile is not player's own
            const dx = p.x - player.x;
            const dy = p.y - player.y;
            if (dx*dx + dy*dy < 10*10) { // Player radius = 10
                 player.health = Math.max(0, player.health - 0.1);
                 console.log(`Player hit! Health: ${player.health.toFixed(1)}`);
                 return false; // Projectile is destroyed
            }
        }
        // Check collision with bots
        bots.forEach(bot => {
             if (p.ownerId === undefined || p.ownerId !== bot.id) { // Check if not bot's own
                 const dx = p.x - bot.x;
                 const dy = p.y - bot.y;
                 if (dx*dx + dy*dy < 10*10) { // Bot radius = 10
                     bot.health = Math.max(0, bot.health - 0.1);
                     console.log(`Bot ${bot.id} hit! Health: ${bot.health.toFixed(1)}`);
                     // If bot health is 0, remove it?
                     // TODO: Remove defeated bots
                     p.life = 0; // Mark projectile for removal
                 }
             }
        });


        return p.life > 0 && p.x > 0 && p.x < WORLD_WIDTH && p.y > 0 && p.y < WORLD_HEIGHT;
    });

    // --- Player Interaction ---
    if (player.isInteracting) {
         ammoItems.forEach((item, index) => {
            const dx = item.x - player.x;
            const dy = item.y - player.y;
            if (dx * dx + dy * dy < (item.size + 10) * (item.size + 10)) { // Player radius = 10
                player.ammo = Math.min(MAX_AMMO, player.ammo + 5);
                ammoItems.splice(index, 1);
                spawnAmmo(); // Respawn one
                console.log(`Player picked up ammo. Ammo: ${player.ammo}`);
            }
        });
        player.isInteracting = false; // Interaction is momentary
    }

     // --- Remove defeated bots ---
     bots = bots.filter(bot => {
         if (bot.health <= 0) {
             console.log(`Bot ${bot.id} defeated.`);
             if (bot.hierarchy) bot.hierarchy.dispose(); // Clean up TF resources
             return false;
         }
         return true;
     });


}

// --- Hierarchical Processing (Async) ---
async function processHierarchies() {
    // --- Player Hierarchy ---
    const playerSenses = getSenses(player);
    // Combine actions (inputs) and senses
    const currentPlayerInputData = { ...inputs, ...playerSenses };

    // Train based on the *actual* current state (S(t+1)) vs the *last* prediction (P(t+1))
    // Note: trainTick internally uses the input that *caused* the last prediction
    if (playerHierarchy.lastPredictionTensor) {
        await playerHierarchy.trainTick(currentPlayerInputData);
    }


    // Process current state S(t+1) to get next prediction P(t+2) and anomaly
    const playerResult = await playerHierarchy.processTick(currentPlayerInputData);

    // Update player status display
    playerStatusDiv.innerHTML = `
        Anomaly: ${playerResult.currentAnomaly.toFixed(5)} <br>
        Skill: ${playerResult.skillLevel.toFixed(3)} <br>
        Prediction (MoveY): ${playerResult.predictionObject?.moveY?.toFixed(2)} <br>
        Prediction (Attack): ${playerResult.predictionObject?.attack?.toFixed(2)}
    `;

// Check for bot spawning
    if (playerResult.spawnBotSignal) {
        console.log("Spawn Bot Signal Received!");
        const botId = nextBotId++;
        const newBot = {
            id: botId,
            x: Math.random() * WORLD_WIDTH,
            y: Math.random() * WORLD_HEIGHT,
            angle: Math.random() * 2 * Math.PI,
            ammo: 5,
            health: 1.0,
            hierarchy: new CorticalHierarchy(config), // Create new hierarchy instance
            lastInputData: null,
            isAttacking: false,
        };

        // Clone weights from player
        const playerWeights = playerHierarchy.getWeights();
        newBot.hierarchy.setWeights(playerWeights); // This clones the weights for the bot

        // DO NOT DISPOSE THE ORIGINAL PLAYER WEIGHTS HERE
        // playerWeights.forEach(w => w.dispose()); // <-- REMOVE THIS LINE

        // The player's hierarchy still needs its weights.
        // The bot's hierarchy has its own clones thanks to setWeights.

        bots.push(newBot);
        console.log(`Bot ${botId} spawned.`);

        // Note: Technically, the tensors returned by getWeights() aren't marked
        // with tf.keep(). While usually safe for model weights, if you were paranoid
        // you could clone them immediately *before* passing to setWeights,
        // but the current approach where setWeights handles cloning is fine and standard.
        // Just *never* dispose the result of model.getWeights() unless you are
        // disposing the entire model.
    }

    // --- Bot Hierarchies ---
    let botStatusHTML = "--- Bots ---<br>";
    for (const bot of bots) {
        const botSenses = getSenses(bot);
        // Bot "actions" are part of its state from the *last* prediction
        // We need to reconstruct the bot's input vector from its prediction + senses
        // Or simpler: Bot's input is just its *senses* for now. It learns to predict future senses based on past senses.
        // To make it act, we use the action *part* of the prediction vector.
        // Let's refine this: Bot input should include *what it tried to do* last tick + current senses.

        // Get the action part from the *last* prediction made by the bot
        const lastPrediction = bot.hierarchy.getPredictionObject() || {}; // Use empty if first tick
        const botActions = {
             moveX: lastPrediction.moveX ?? 0.5, // Default to neutral if no prediction yet
             moveY: lastPrediction.moveY ?? 0.5,
             lookX: lastPrediction.lookX ?? 0.5,
             lookY: lastPrediction.lookY ?? 0.5, // Not used in 2D but kept for consistency
             jump: lastPrediction.jump ?? 0,
             attack: lastPrediction.attack ?? 0,
             interact: lastPrediction.interact ?? 0,
             run: lastPrediction.run ?? 0,
        };

        const currentBotInputData = { ...botActions, ...botSenses };
        bot.lastInputData = currentBotInputData; // Store for next tick maybe?

        // Train bot based on current state vs its last prediction
        if (bot.hierarchy.lastPredictionTensor) {
            await bot.hierarchy.trainTick(currentBotInputData);
        }

        // Process bot's current state to predict its next state/actions
        const botResult = await bot.hierarchy.processTick(currentBotInputData);

         botStatusHTML += `
            Bot ${bot.id} | Skill: ${botResult.skillLevel.toFixed(3)} | Anom: ${botResult.currentAnomaly.toFixed(5)} <br>
            &nbsp;&nbsp; Pred MoveY: ${botResult.predictionObject?.moveY?.toFixed(2)} | Pred Attack: ${botResult.predictionObject?.attack?.toFixed(2)} <br>
         `;
    }
     botStatusDiv.innerHTML = botStatusHTML;

    // Optional Logging with Danfo.js
    // frameCount++;
    // if (frameCount % logInterval === 0 && typeof dfd !== 'undefined') {
    //     logData.push({ timestamp: performance.now(), ...currentPlayerInputData, anomaly: playerResult.currentAnomaly, skill: playerResult.skillLevel });
    //     // Can create DataFrame later: new dfd.DataFrame(logData).print();
    // }

}

// --- Rendering ---
function draw() {
    // Clear canvas
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, WORLD_WIDTH, WORLD_HEIGHT);

    // Draw ammo items
    ctx.fillStyle = 'yellow';
    ammoItems.forEach(item => {
        ctx.beginPath();
        ctx.arc(item.x, item.y, item.size, 0, Math.PI * 2);
        ctx.fill();
    });

    // Draw projectiles
    ctx.fillStyle = 'red';
    projectiles.forEach(p => {
         ctx.beginPath();
         ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
         ctx.fill();
    });

    // Draw player
    ctx.save();
    ctx.translate(player.x, player.y);
    ctx.rotate(player.angle);
    ctx.fillStyle = botViewActive ? 'gray' : 'lime'; // Dim player if viewing bot
    ctx.beginPath();
    ctx.moveTo(10, 0); // Nose of triangle
    ctx.lineTo(-5, -7);
    ctx.lineTo(-5, 7);
    ctx.closePath();
    ctx.fill();
    // Health bar
    ctx.fillStyle = 'red';
    ctx.fillRect(-10, -15, 20, 3);
    ctx.fillStyle = 'green';
    ctx.fillRect(-10, -15, 20 * player.health, 3);
    ctx.restore();


    // Draw bots
    bots.forEach(bot => {
        ctx.save();
        ctx.translate(bot.x, bot.y);
        ctx.rotate(bot.angle);
        ctx.fillStyle = (botViewActive && bot.id === targetBotId) ? 'cyan' : 'blue'; // Highlight target bot
        ctx.beginPath();
        ctx.moveTo(10, 0); // Triangle shape
        ctx.lineTo(-5, -7);
        ctx.lineTo(-5, 7);
        ctx.closePath();
        ctx.fill();
        // Health bar
        ctx.fillStyle = 'red';
        ctx.fillRect(-10, -15, 20, 3);
        ctx.fillStyle = 'green';
        ctx.fillRect(-10, -15, 20 * bot.health, 3);
        ctx.restore();
    });
}

// --- Game Loop ---
let frameCounter = 0; // Use existing frameCount or add a new one
const HIERARCHY_PROCESS_INTERVAL = 3; // Process every 3 frames (adjust as needed)
let hierarchyProcessingScheduled = false;

async function gameLoop(timestamp) {
    if (!gameRunning) return;

    const dt = Math.min(0.05, (timestamp - lastTimestamp) / 1000); // Delta time in seconds, capped
    const dtFactor = dt * 60; // Factor to adjust speeds if they were tuned for 60fps
    lastTimestamp = timestamp;

updateInputs(); // Update keyboard/mouse/touch inputs state

    // Update game logic (physics, basic AI reactions)
    update(dt);

    // Process Hierarchies (less frequently)
    frameCounter++;
    if (frameCounter % HIERARCHY_PROCESS_INTERVAL === 0 && !hierarchyProcessingScheduled) {
        hierarchyProcessingScheduled = true;

        // Use requestIdleCallback or setTimeout to yield slightly, preventing
        // complete blocking, but still ensure it runs reasonably soon.
        // setTimeout is simpler and more reliable across browsers.
        setTimeout(async () => {
            try {
                await processHierarchies(); // Run the async processing
            } catch (error) {
                 console.error("Error during scheduled hierarchy processing:", error);
            } finally {
                hierarchyProcessingScheduled = false; // Allow scheduling again
            }
        }, 0); // Timeout 0 yields to other tasks like rendering first
    }


    // Render the current state (always do this every frame)
    draw();


    // Check for game end condition? (e.g., player health <= 0)

    // Request next frame
    requestAnimationFrame(gameLoop);
}

// --- Start ---
// Ensure TF is ready before initializing
tf.ready().then(init);
