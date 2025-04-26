/**
 * corticalHierarchy.js
 *
 * A library inspired by spatiotemporal cortical hierarchies for learning
 * and prediction in dynamic environments like games. It treats actions
 * and senses as a unified stream, learns patterns, predicts future states,
 * calculates anomalies, and can spawn 'bots' based on learned models.
 *
 * Uses TensorFlow.js (tfjs)
 */

// Ensure tfjs is loaded before this script

class CorticalHierarchy {
    /**
     * @param {object} config Configuration object
     * @param {object} config.streamConfig Defines input streams: { streamName: size, ... }. Sizes must be numbers.
     * @param {number[]} config.hiddenLayers Array defining units in hidden layers, e.g., [64, 32].
     * @param {number} [config.learningRate=0.001] Learning rate for the optimizer.
     * @param {number} [config.skillThreshold=0.95] Skill level threshold to trigger bot spawn signal.
     * @param {number} [config.skillHistoryLength=100] Number of ticks to average anomaly for skill.
     * @param {string} [config.activation='relu'] Activation function for hidden layers.
     * @param {string} [config.finalActivation='sigmoid'] Activation for output layer (assumes inputs normalized 0-1).
     */
    constructor(config) {
        console.log("CorticalHierarchy constructor: START"); // *** ADDED ***

        if (!tf) {
            console.error("CorticalHierarchy constructor: FATAL - tf object not found!"); // *** ADDED ***
            throw new Error("TensorFlow.js (tf) not found. Load it before CorticalHierarchy.");
        }

        this.config = {
            learningRate: 0.001,
            skillThreshold: 0.95, // Example: 1 - avg_anomaly > threshold
            skillHistoryLength: 100,
            activation: 'relu',
            finalActivation: 'sigmoid', // Use 'tanh' if normalizing to -1 to 1
            ...config
        };
         console.log("CorticalHierarchy constructor: Config processed", this.config); // *** ADDED ***

        if (!this.config.streamConfig || Object.keys(this.config.streamConfig).length === 0) {
            throw new Error("config.streamConfig is required and must not be empty.");
        }
        if (!this.config.hiddenLayers || !Array.isArray(this.config.hiddenLayers)) {
            throw new Error("config.hiddenLayers is required and must be an array.");
        }

        this.streamNames = Object.keys(this.config.streamConfig);
        this.streamSizes = Object.values(this.config.streamConfig);
        this.inputSize = this.streamSizes.reduce((sum, size) => sum + size, 0);
        this.outputSize = this.inputSize; // Predicts the next state of the same streams
        console.log(`CorticalHierarchy constructor: Input/Output size calculated = ${this.inputSize}`); // *** ADDED ***

        this.streamMapping = {};
        let currentIndex = 0;
        for (const name of this.streamNames) {
            const size = this.config.streamConfig[name];
            this.streamMapping[name] = { start: currentIndex, end: currentIndex + size };
            currentIndex += size;
        }

        // --- Optimizer Creation Logging ---
        console.log("CorticalHierarchy constructor: Checking tf.train..."); // *** ADDED ***
        if (!tf.train) {
            console.error("CorticalHierarchy constructor: FATAL - tf.train module not found!"); // *** ADDED ***
            throw new Error("tf.train is not available!");
        }
        console.log("CorticalHierarchy constructor: tf.train found:", tf.train); // *** ADDED ***

        console.log(`CorticalHierarchy constructor: Attempting to create optimizer with learning rate: ${this.config.learningRate}`); // *** ADDED ***
        let createdOptimizer;
        try {
            // Check if learning rate is valid before passing
            if (typeof this.config.learningRate !== 'number' || !isFinite(this.config.learningRate)) {
                 console.error(`CorticalHierarchy constructor: Invalid learning rate: ${this.config.learningRate}`); // *** ADDED ***
                 throw new Error(`Invalid learning rate: ${this.config.learningRate}`);
            }
            createdOptimizer = tf.train.adam(this.config.learningRate);
            console.log("CorticalHierarchy constructor: Optimizer creation attempted. Result:", createdOptimizer); // *** ADDED ***
            if (!createdOptimizer) {
                 console.error("CorticalHierarchy constructor: tf.train.adam() returned undefined/null/falsy!"); // *** ADDED ***
            }
        } catch (error) {
            console.error("CorticalHierarchy constructor: ERROR during tf.train.adam():", error); // *** ADDED ***
            throw error; // Re-throw to ensure failure is visible
        }
        // --- End Optimizer Logging ---

        this.optimizer = createdOptimizer;
        console.log("CorticalHierarchy constructor: Assigned this.optimizer:", this.optimizer); // *** ADDED ***

        // Initialize other properties BEFORE building model
        this.lastPredictionTensor = null;
        this.lastPredictionObject = null;
        this.lastInputTensor = null; // Added initialization
        this.currentAnomaly = 0.0;
        this.anomalyHistory = [];
        this.skillLevel = 0.0;
        this.consecutiveLowAnomalyTicks = 0;
        this.spawnBotSignal = false;
        this.weightsReceived = false;

        console.log("CorticalHierarchy constructor: State variables initialized."); // *** ADDED ***
        console.log("CorticalHierarchy constructor: Calling _buildModel..."); // *** ADDED ***
        this.model = this._buildModel(); // Now call buildModel
        console.log("CorticalHierarchy constructor: _buildModel returned."); // *** ADDED ***

        console.log(`CorticalHierarchy initialized fully.`); // *** MODIFIED ***
    }

    /** Builds the sequential TF.js model */
    _buildModel() {
        console.log("_buildModel: START"); // *** ADDED ***
        const model = tf.sequential();
        const inputShape = [this.inputSize];

        // Add hidden layers
        let prevLayerSize = this.inputSize;
        this.config.hiddenLayers.forEach((units, i) => {
             console.log(`_buildModel: Adding hidden layer ${i} with ${units} units.`); // *** ADDED ***
            model.add(tf.layers.dense({
                units: units,
                inputShape: i === 0 ? inputShape : undefined,
                activation: this.config.activation,
                name: `hidden_${i}`
            }));
            prevLayerSize = units;
        });

        // Add output layer
        console.log(`_buildModel: Adding output layer with ${this.outputSize} units.`); // *** ADDED ***
        model.add(tf.layers.dense({
            units: this.outputSize,
            activation: this.config.finalActivation, // Assumes input is normalized e.g., 0-1
            name: 'output'
        }));
        console.log("_buildModel: Layers added."); // *** ADDED ***

        console.log("_buildModel: About to compile. Checking this.optimizer:", this.optimizer); // *** MODIFIED ***
        if (!this.optimizer) {
            console.error("_buildModel: ERROR - this.optimizer is undefined or falsy before compile!"); // *** ADDED ***
             // Optionally throw an error here to stop execution clearly
             throw new Error("_buildModel Error: Optimizer is missing before compile!");
        } else {
             console.log("_buildModel: Optimizer seems valid, proceeding to compile."); // *** ADDED ***
        }

        try {
             console.log("_buildModel: Calling model.compile..."); // *** ADDED ***
             model.compile({ optimizer: this.optimizer, loss: 'meanSquaredError' });
             console.log("_buildModel: model.compile finished successfully."); // *** ADDED ***
        } catch (compileError) {
             console.error("_buildModel: ERROR during model.compile():", compileError); // *** ADDED ***
             console.error("_buildModel: Optimizer object during error was:", this.optimizer); // *** ADDED ***
             throw compileError; // Re-throw
        }


        console.log("_buildModel: Model summary follows:"); // *** ADDED ***
        model.summary(); // Keep summary
        console.log("_buildModel: FINISHED"); // *** ADDED ***
        return model;
    }

    /**
     * Converts an input object { streamName: value | array, ... } to a flat tensor.
     * Assumes values are already normalized between 0 and 1 (or -1 and 1 if using tanh).
     */
    _vectorizeInput(inputData) {
        return tf.tidy(() => {
            const buffer = tf.buffer([1, this.inputSize]);
            for (const streamName of this.streamNames) {
                const mapping = this.streamMapping[streamName];
                const value = inputData[streamName];

                if (value === undefined) {
                    console.warn(`Stream "${streamName}" missing in inputData. Filling with 0.`);
                    for (let i = mapping.start; i < mapping.end; ++i) {
                        buffer.set(0, 0, i);
                    }
                    continue;
                }

                if (Array.isArray(value)) {
                    if (value.length !== (mapping.end - mapping.start)) {
                       console.warn(`Stream "${streamName}" size mismatch. Expected ${mapping.end - mapping.start}, got ${value.length}.`);
                    }
                    for (let i = 0; i < (mapping.end - mapping.start); ++i) {
                         // Clamp or normalize values if needed, here assuming they are correct
                        buffer.set(value[i] !== undefined ? value[i] : 0, 0, mapping.start + i);
                    }
                } else { // Single value stream
                     if (mapping.end - mapping.start !== 1) {
                         console.warn(`Stream "${streamName}" expects size 1, got single value.`);
                     }
                    buffer.set(value, 0, mapping.start);
                }
            }
            return buffer.toTensor();
        });
    }

     /** Converts an output tensor back to a { streamName: value | array, ... } object */
    async _devectorizeOutput(tensor) {
        const data = await tensor.data(); // Get tensor data asynchronously
        const outputData = {};
        for (const streamName of this.streamNames) {
            const mapping = this.streamMapping[streamName];
            const size = mapping.end - mapping.start;
            if (size === 1) {
                outputData[streamName] = data[mapping.start];
            } else {
                outputData[streamName] = Array.from(data.slice(mapping.start, mapping.end));
            }
        }
        return outputData;
    }

    /**
     * Process the current state S(t) to predict the next state P(t+1).
     * Also calculates anomaly based on the *previous* prediction P(t) and current state S(t).
     * IMPORTANT: This function now RETURNS the prediction P(t+1) but DOES NOT train.
     * Training happens in trainTick() using the *next* state S(t+1).
     *
     * @param {object} currentInputData - The current state S(t) as { streamName: value, ... }
     * @returns {Promise<object>} A promise resolving to { predictionObject, currentAnomaly, spawnBotSignal }
     */
    async processTick(currentInputData) {
        this.spawnBotSignal = false; // Reset signal each tick

        const St_Tensor = this._vectorizeInput(currentInputData); // S(t)

        let anomaly = 0.0;
        if (this.lastPredictionTensor) {
            // Calculate anomaly: Difference between previous prediction P(t) and current actual state S(t)
            const errorTensor = tf.sub(this.lastPredictionTensor, St_Tensor);
            const mse = tf.mean(tf.square(errorTensor)).dataSync()[0];
            anomaly = mse; // Using Mean Squared Error as anomaly score
            errorTensor.dispose();

             // Update anomaly history and skill
            this.anomalyHistory.push(anomaly);
            if (this.anomalyHistory.length > this.config.skillHistoryLength) {
                this.anomalyHistory.shift(); // Keep history windowed
            }
            const avgAnomaly = this.anomalyHistory.reduce((a, b) => a + b, 0) / this.anomalyHistory.length;
            // Skill: Higher skill = lower anomaly. Clamp between 0 and 1.
            this.skillLevel = Math.max(0, Math.min(1, 1.0 - Math.sqrt(avgAnomaly))); // Use sqrt to make it less sensitive to small errors
            
            console.log(`Tick Anomaly: ${anomaly.toFixed(5)}, Avg Anomaly: ${avgAnomaly.toFixed(5)}, Skill Level: ${this.skillLevel.toFixed(3)}`);

            // Check for bot spawn condition
            if (this.weightsReceived) { // Only spawn if weights are established (trained or loaded)
            console.log(`Checking spawn: weightsReceived=${this.weightsReceived}, skill=${this.skillLevel.toFixed(3)} > threshold=${this.config.skillThreshold}?`); // <-- ADD 1
                 if (this.skillLevel > this.config.skillThreshold) {
                    this.consecutiveLowAnomalyTicks++;
                    console.log(`Skill above threshold. Consecutive Ticks: ${this.consecutiveLowAnomalyTicks}`); // <-- ADD 2
                 } else {
                    if (this.consecutiveLowAnomalyTicks > 0) { // Log reset only if it was counting
                     console.log(`Skill below threshold. Resetting consecutive ticks from ${this.consecutiveLowAnomalyTicks}`); // <-- ADD 3
                     this.consecutiveLowAnomalyTicks = 0; // Reset counter if skill drops
                }
                 }

                 // Require a certain number of consecutive ticks above threshold
                 // Adjust '10' as needed for stability
const requiredTicks = 10; // Define required ticks clearly
             console.log(`Comparing consecutive ticks ${this.consecutiveLowAnomalyTicks} >= ${requiredTicks}`); // <-- ADD 4
             if (this.consecutiveLowAnomalyTicks >= requiredTicks) {
                 this.spawnBotSignal = true;
                 console.log(`>>> SPAWN SIGNAL SET TRUE <<< Resetting consecutive ticks.`); // <-- ADD 5
                 this.consecutiveLowAnomalyTicks = 0; // Reset after signaling
             }
        } else {
             console.log("Checking spawn: weightsReceived=false. No spawn check."); // <-- ADD 6
        }


        } else {
             // If no previous prediction, initial anomaly is 0 or undefined
             this.skillLevel = 0.0; // Start with zero skill
        }
        this.currentAnomaly = anomaly;


        // Predict the NEXT state P(t+1) based on current state S(t)
        const Pt1_Tensor = tf.tidy(() => this.model.predict(St_Tensor)); // P(t+1)

        // Clean up previous prediction tensor BEFORE assigning new one
        if (this.lastPredictionTensor) {
            tf.dispose(this.lastPredictionTensor);
        }
        // Store the new prediction P(t+1) for the *next* tick's anomaly calculation
        this.lastPredictionTensor = tf.keep(Pt1_Tensor.clone());

        // Prepare results (don't block on devectorize here)
        const predictionObjectPromise = this._devectorizeOutput(Pt1_Tensor);

        // Clean up S(t) tensor, keep P(t+1) stored in this.lastPredictionTensor
        St_Tensor.dispose();
        Pt1_Tensor.dispose(); // Dispose the one returned by predict, we kept a clone

        // Await the devectorized object before returning
        this.lastPredictionObject = await predictionObjectPromise;

        return {
            predictionObject: this.lastPredictionObject,
            currentAnomaly: this.currentAnomaly,
            skillLevel: this.skillLevel,
            spawnBotSignal: this.spawnBotSignal
        };
    }

    /**
     * Train the model based on the actual outcome S(t+1) of the previous prediction P(t+1).
     * This should be called *after* processTick, providing the *next* frame's actual data.
     * @param {object} actualNextInputData - The actual state S(t+1) corresponding to the last prediction P(t+1).
     * @returns {Promise<number|null>} Loss value after training, or null if no training occurred.
     */
    async trainTick(actualNextInputData) {
        if (!this.lastPredictionTensor) {
            // Cannot train if we haven't made a prediction yet
            return null;
        }

        const St1_Tensor = this._vectorizeInput(actualNextInputData); // S(t+1) - the target

        try {
            // Train the model: Predict P(t+1) again from S(t) implicitly via fit
            // We need the S(t) that *led* to P(t+1) for training.
            // This requires storing S(t) from the *previous* processTick call.
            // Let's modify state management slightly:

            // --- Conceptual Change ---
            // processTick(S(t)) -> returns P(t+1) prediction
            // trainTick(S(t), S(t+1)) -> uses S(t) as input, S(t+1) as target
            // This is cleaner but requires passing S(t) to trainTick.

            // --- Alternative (more stateful) ---
            // Let's try to keep it simpler: We train the model to map S(t) -> S(t+1).
            // The `processTick` predicts P(t+1) from S(t).
            // The `trainTick` uses S(t+1) as the target for the prediction P(t+1) that was made.
            // This seems less direct for standard supervised learning `fit`.

            // --- Let's use model.fit ---
            // We need the input S(t) that generated the prediction P(t+1).
            // We only have S(t+1) available here.
            // This implies we need to store the *input* tensor from the previous tick as well.

            // --- Redesigning State Management ---
            // Let's store the *input* tensor that generated the *last prediction*.
            if (this.lastInputTensor) {
                 // Target is the current actual state S(t+1)
                 const targetTensor = St1_Tensor;

                 // Input is the state S(t) that led to the prediction P(t+1)
                 const inputTensor = this.lastInputTensor;

                 // Perform training step
                 const history = await this.model.fit(inputTensor, targetTensor, {
                     epochs: 1,
                     batchSize: 1,
                     verbose: 0 // 0 = silent, 1 = progress bar
                 });

                 const loss = history.history.loss[0];
                 // console.log("Train loss:", loss); // Optional logging

                 // Mark that weights have been influenced by data
                 this.weightsReceived = true;

                 // Clean up the input tensor used for training BEFORE storing the next one
                 tf.dispose(this.lastInputTensor);
                 this.lastInputTensor = tf.keep(St1_Tensor.clone()); // Store S(t+1) for the *next* training step

                 targetTensor.dispose(); // St1_Tensor is cloned into lastInputTensor or disposed below

                 return loss;

            } else {
                 // First tick, just store the current input S(t+1) for the next training step
                 this.lastInputTensor = tf.keep(St1_Tensor.clone());
                 return null; // No training happened
            }

        } catch (error) {
            console.error("Training failed:", error);
            return null;
        } finally {
            // Ensure St1_Tensor is disposed if not kept
            if (this.lastInputTensor !== St1_Tensor) {
                 St1_Tensor.dispose();
            }
        }
    }


    /** Get current skill level */
    getSkillLevel() {
        return this.skillLevel;
    }

    /** Get latest anomaly score */
    getAnomalies() {
        // In V1, we only have one main anomaly score (prediction error)
        return { overall: this.currentAnomaly };
    }

    /** Get the last predicted state object */
    getPredictionObject() {
        return this.lastPredictionObject;
    }

    /** Get model weights for saving/cloning */
    getWeights() {
        return this.model.getWeights();
    }

    /**
     * Set model weights (e.g., for loading or cloning bots).
     * Weights should be an array of Tensors.
     */
    setWeights(weights) {
        if (weights && Array.isArray(weights) && weights.length > 0) {
            this.model.setWeights(weights.map(w => tf.keep(w.clone()))); // Clone to avoid sharing tensors
             this.weightsReceived = true; // Mark weights as loaded/set
            console.log("Weights set successfully.");
        } else {
            console.warn("setWeights received invalid input.");
        }
    }

    /** Clean up TF.js resources */
    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        if (this.lastPredictionTensor) {
            tf.dispose(this.lastPredictionTensor);
        }
        if (this.lastInputTensor) {
             tf.dispose(this.lastInputTensor);
        }
        // Dispose weights potentially passed via setWeights? Risky if shared.
        // Assume weights passed in are managed elsewhere or clones are safe.
        console.log("CorticalHierarchy disposed.");
    }
}

// If running in Node.js, uncomment the following line:
// module.exports = CorticalHierarchy;
