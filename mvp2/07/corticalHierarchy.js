// corticalHierarchy.js

// Ensure tfjs is loaded before this script, e.g., via CDN in HTML.
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>

class CorticalHierarchy {
    constructor(config = {}) {
        this.numLevels = config.numLevels || 3; // e.g., Input -> Hidden1 -> HiddenTop
        this.learningRate = config.learningRate || 0.01;
        this.streamNames = config.streamNames || []; // Must be set via defineStreams
        this.streamSize = 0; // Determined by streamNames length
        this.layers = [];
        this.optimizers = [];
        this.layerLatentSizes = config.layerLatentSizes || this._getDefaultLatentSizes(); // e.g., [inputSize, 16, 8]
        this.isInitialized = false;

        // Anomaly tracking
        this.anomalyThreshold = config.anomalyThreshold || 0.1; // Example threshold
        this.anomalyHistorySize = config.anomalyHistorySize || 100; // Number of ticks to average over
        this.anomalyHistory = Array(this.numLevels).fill(0).map(() => []);
        this.averageAnomalies = Array(this.numLevels).fill(Infinity);

        // Bot spawning logic
        this.botSpawnPatience = config.botSpawnPatience || 200; // Ticks of low anomaly required
        this.botSpawnCounter = 0;
        this.skillLevel = 0; // Track skill progression
        this.readyToSpawnBot = false;

        console.log("CorticalHierarchy created. Call defineStreams() and initializeModels() before use.");
    }

    _getDefaultLatentSizes() {
        const sizes = [this.streamSize]; // Start with input size
        let currentSize = this.streamSize;
        for (let i = 1; i < this.numLevels; i++) {
            // Simple heuristic: reduce size, but not too drastically
            currentSize = Math.max(4, Math.ceil(currentSize * 0.6));
            sizes.push(currentSize);
        }
        console.log("Using default latent sizes:", sizes);
        return sizes;
    }

    defineStreams(streamNames) {
        if (this.isInitialized) {
            console.error("Cannot redefine streams after initialization.");
            return;
        }
        this.streamNames = [...streamNames];
        this.streamSize = this.streamNames.length;
        this.layerLatentSizes[0] = this.streamSize; // Input layer size is stream size
        if(this.layerLatentSizes.length !== this.numLevels) {
             console.warn("Mismatch between numLevels and layerLatentSizes length. Adjusting latent sizes.");
             this.layerLatentSizes = this._getDefaultLatentSizes(); // Recalculate if needed
             this.layerLatentSizes[0] = this.streamSize;
        }
        console.log(`Streams defined: ${this.streamSize} streams`, this.streamNames);
        console.log(`Layer latent sizes:`, this.layerLatentSizes);
    }

async initializeModels() {
        // ... (previous checks) ...
        console.log("Initializing models...");

        this.layers = [];
        // this.optimizers = []; // Remove single optimizer array

        for (let i = 0; i < this.numLevels; i++) {
            const inputSize = this.layerLatentSizes[i];
            const forwardOutputSize = (i < this.numLevels - 1) ? this.layerLatentSizes[i + 1] : inputSize;
            const backwardOutputSize = inputSize;
            const backwardInputSize = (i < this.numLevels - 1) ? this.layerLatentSizes[i + 1] : inputSize; // Corrected: Input for backward model comes from layer above

            // --- Forward Model (Encoder) ---
            const forwardModel = tf.sequential();
            forwardModel.add(tf.layers.dense({
                units: Math.max(4, Math.ceil((inputSize + forwardOutputSize) / 2)),
                inputShape: [inputSize],
                activation: 'relu',
                name: `L${i}_Fwd_Hidden` // Add names for easier debugging
            }));
            forwardModel.add(tf.layers.dense({
                units: forwardOutputSize,
                activation: 'tanh',
                 name: `L${i}_Fwd_Output`
            }));

            // --- Backward Model (Predictor/Decoder) ---
            const backwardModel = tf.sequential();
            backwardModel.add(tf.layers.dense({
                units: Math.max(4, Math.ceil((backwardInputSize + backwardOutputSize) / 2)),
                inputShape: [backwardInputSize],
                activation: 'relu',
                name: `L${i}_Bwd_Hidden`
            }));
            backwardModel.add(tf.layers.dense({
                units: backwardOutputSize,
                activation: 'linear', // Or match normalization range if not 0-1 / -1-1
                name: `L${i}_Bwd_Output`
            }));

            // --- Create SEPARATE Optimizers ---
            const forwardOptimizer = tf.train.adam(this.learningRate);
            const backwardOptimizer = tf.train.adam(this.learningRate);

            this.layers.push({
                level: i,
                forwardModel: forwardModel,
                backwardModel: backwardModel,
                forwardOptimizer: forwardOptimizer,   // Store forward optimizer
                backwardOptimizer: backwardOptimizer, // Store backward optimizer
                inputSize: inputSize,
                forwardOutputSize: forwardOutputSize,
                backwardInputSize: backwardInputSize,
            });

            // this.optimizers.push(tf.train.adam(this.learningRate)); // Remove this
        }

        this.isInitialized = true;
        console.log("Models initialized successfully with separate optimizers.");
        await tf.ready();
        console.log("TensorFlow.js backend ready:", tf.getBackend());
    }
    
    _dataToObject(tensor) {
        if (!tensor || typeof tensor.arraySync !== 'function') {
            console.error("Invalid tensor provided to _dataToObject");
            return {};
        }
         const values = tensor.arraySync()[0]; // Assuming batch size 1
         const obj = {};
         this.streamNames.forEach((name, i) => {
             obj[name] = values[i];
         });
         return obj;
    }

    _objectToDataTensor(dataObject) {
        const tensorArray = this.streamNames.map(name => dataObject[name] ?? 0); // Default missing values to 0
        // Create a 2D tensor [1, streamSize] (batch size 1)
        return tf.tensor2d([tensorArray]);
    }

    processTick(currentData) {
        if (!this.isInitialized) {
            console.error("Hierarchy not initialized. Call defineStreams() and initializeModels() first.");
            return { anomalies: [], prediction: {}, rawPredictionTensor: null };
        }

        // tf.tidy will manage intermediate tensors. Tensors returned are kept.
        const { anomalies, predictionTensor, actualInputTensors, predictedInputTensors } = tf.tidy(() => {
            const currentDataTensor = this._objectToDataTensor(currentData); // Shape [1, streamSize]

            let forwardActivations = [];
            let currentForwardInput = currentDataTensor;
            forwardActivations.push(currentForwardInput); // Actual L0 input

            // 1. Forward Pass (Encoding Upwards)
            for (let i = 0; i < this.numLevels; i++) {
                const layer = this.layers[i];
                // The input for the forward model is the activation from the layer below (or the raw input for L0)
                const currentForwardModelInput = forwardActivations[i];

                if (i < this.numLevels - 1) { // If not the top layer, run forward model
                    const output = layer.forwardModel.predict(currentForwardModelInput);
                    forwardActivations.push(output); // Store output (becomes input for next layer's forward/backward)
                } else {
                    // Top level: its "output" is its input representation. Store it so array lengths match.
                    forwardActivations.push(currentForwardModelInput.clone()); // Clone to ensure distinct tensor if needed later
                }
            }
             // Check length: should be numLevels + 1 (input + output of each layer)
             // console.log("Forward Activations length:", forwardActivations.length);


            // 2. Backward Pass (Prediction Downwards)
             let predictedInputs = [];
             // Input for the top layer's backward model is its own activation/output
             let currentBackwardInput = forwardActivations[this.numLevels]; // Activation output *from* the top layer processing

             for (let i = this.numLevels - 1; i >= 0; i--) {
                 const layer = this.layers[i];
                 const prediction = layer.backwardModel.predict(currentBackwardInput); // Predict representation for layer i
                 predictedInputs.unshift(prediction); // Add prediction to the beginning [predL0, predL1, ...]

                 // The input for the next lower layer's backward model (predicting layer i-1)
                 // comes from the forward activation of the current layer (i)
                 if (i > 0) {
                     currentBackwardInput = forwardActivations[i]; // Use activation *output* of layer i
                 }
                 // When i=0, loop terminates. We have predictedInputs[0] which is prediction for L0.
             }
             // Check length: should be numLevels
             // console.log("Predicted Inputs length:", predictedInputs.length);


            // 3. Anomaly Calculation
            let tickAnomalies = [];
            for (let i = 0; i < this.numLevels; i++) {
                 const actual = forwardActivations[i];    // Actual input received by layer i
                 const predicted = predictedInputs[i];  // Prediction *for* the input of layer i

                 if (!actual || !predicted) {
                     console.error(`Missing tensor for anomaly calculation at level ${i}. Actual: ${actual}, Predicted: ${predicted}`);
                      tickAnomalies.push(1.0); // Assign high anomaly if tensors are missing
                     continue;
                 }

                 const anomalyTensor = tf.losses.meanSquaredError(actual, predicted);
                 const anomaly = anomalyTensor.dataSync()[0];
                 tickAnomalies.push(isNaN(anomaly) ? 1.0 : anomaly);
                 // anomalyTensor is disposed automatically by tf.tidy
            }

            // Return the tensors needed outside the tidy scope
            return {
                anomalies: tickAnomalies,
                predictionTensor: predictedInputs[0], // Prediction for L0
                actualInputTensors: forwardActivations, // All activations [L0_input, L1_input, L2_input, L3_input(top_output)]
                predictedInputTensors: predictedInputs   // All predictions [L0_pred, L1_pred, L2_pred]
            };
        }); // End tf.tidy

        // --- Processing after tidy ---

        // 1. Convert the needed prediction tensor immediately
        const predictionObject = this._dataToObject(predictionTensor);

        // 2. Dispose the predictionTensor now that we have the object
        tf.dispose(predictionTensor);

        // 3. Update anomaly history (uses JS numbers, no tensors involved)
        this.updateAnomalyHistory(anomalies);

        // 4. Trigger asynchronous training step, passing the arrays of *kept* tensors
        //    These tensors will be disposed by the tidy inside _trainStep
        this._trainStep(actualInputTensors, predictedInputTensors);

        // 5. Check bot spawning condition (uses JS numbers)
        this.checkBotSpawnCondition();

        // 6. Return results
        return {
            anomalies: this.averageAnomalies,
            prediction: predictionObject,
            rawAnomalies: anomalies
        };
    }
    
     updateAnomalyHistory(currentAnomalies) {
         for (let i = 0; i < this.numLevels; i++) {
             this.anomalyHistory[i].push(currentAnomalies[i]);
             if (this.anomalyHistory[i].length > this.anomalyHistorySize) {
                 this.anomalyHistory[i].shift(); // Remove oldest
             }
             // Calculate average anomaly for the level
             const sum = this.anomalyHistory[i].reduce((a, b) => a + b, 0);
             this.averageAnomalies[i] = sum / this.anomalyHistory[i].length;
         }
     }

_trainStep(actualInputTensors, predictedInputTensors) {
        // ... (previous checks for valid tensors) ...
        if (actualInputTensors[0].isDisposed) {
             // ... (warning and disposal) ...
             return;
         }

        tf.nextFrame().then(() => {
            try {
                tf.tidy(() => { // This tidy manages tensors for this specific training step
                    // console.log("Running async train step...");
                    for (let i = 0; i < this.numLevels; i++) {
                        const layer = this.layers[i];
                        // Get the specific optimizers for this layer
                        const forwardOptimizer = layer.forwardOptimizer;
                        const backwardOptimizer = layer.backwardOptimizer;

                        // --- Train Forward Model ---
                        if (i < this.numLevels - 1) {
                            const forwardModelInput = actualInputTensors[i];
                            const targetForwardOutput = actualInputTensors[i+1];

                            if (!forwardModelInput || forwardModelInput.isDisposed || !targetForwardOutput || targetForwardOutput.isDisposed) {
                                console.warn(`Skipping forward training for level ${i} due to disposed tensors.`);
                                continue;
                            }

                            // Use the forward optimizer
                            forwardOptimizer.minimize(() => { // <--- Use forwardOptimizer
                                const prediction = layer.forwardModel.predict(forwardModelInput);
                                const loss = tf.losses.meanSquaredError(targetForwardOutput, prediction);
                                return loss;
                            });
                        }

                        // --- Train Backward Model ---
                        const backwardModelInput = (i < this.numLevels - 1) ? actualInputTensors[i+1] : actualInputTensors[i];
                        const targetBackwardOutput = actualInputTensors[i];

                         if (!backwardModelInput || backwardModelInput.isDisposed || !targetBackwardOutput || targetBackwardOutput.isDisposed) {
                             console.warn(`Skipping backward training for level ${i} due to disposed tensors.`);
                             continue;
                         }

                        // Use the backward optimizer
                        backwardOptimizer.minimize(() => { // <--- Use backwardOptimizer
                            const prediction = layer.backwardModel.predict(backwardModelInput);
                            const loss = tf.losses.meanSquaredError(targetBackwardOutput, prediction);
                            return loss;
                        });
                    }
                }); // End training tidy
            } catch (error) {
                console.error("Error during async _trainStep:", error);
                actualInputTensors.forEach(t => t && t.dispose && !t.isDisposed && t.dispose());
                predictedInputTensors.forEach(t => t && t.dispose && !t.isDisposed && t.dispose());
            }
            // console.log("Async train step finished");
        });
    }


    getPrediction() {
        // In this design, prediction is generated during processTick.
        // This method could potentially re-run the backward pass if needed separately,
        // but for now, we rely on the last prediction from processTick.
        console.warn("getPrediction() currently relies on the last result from processTick(). Ensure processTick() is called regularly.");
        // Ideally, return the last stored prediction.
        // We need to store the last prediction tensor or object if we want this method.
        // For simplicity, the game loop will get it directly from processTick return value.
        return {}; // Placeholder
    }

    getAnomalyScores() {
        return this.averageAnomalies;
    }

    checkBotSpawnCondition() {
        if (!this.isInitialized) return false;

        // Check if average anomalies across all levels are consistently low
        // Higher levels might be weighted more heavily.
        const overallAnomaly = this.averageAnomalies.reduce((sum, anomaly, i) => sum + anomaly * (i + 1), 0) / this.averageAnomalies.length; // Simple weighted average

        if (this.averageAnomalies.length === this.numLevels && // Ensure history is full
            this.averageAnomalies.every(a => a < this.anomalyThreshold) &&
            this.anomalyHistory[0].length >= this.anomalyHistorySize) // Check history buffer is full
        {
            this.botSpawnCounter++;
            // console.log(`Low anomaly detected. Counter: ${this.botSpawnCounter}/${this.botSpawnPatience}`);
        } else {
            // Reset if anomaly goes up
            if(this.botSpawnCounter > 0) {
                console.log("Anomaly increased, resetting bot spawn counter.");
            }
            this.botSpawnCounter = 0;
            this.readyToSpawnBot = false;
        }

        if (this.botSpawnCounter >= this.botSpawnPatience) {
            if (!this.readyToSpawnBot) {
                console.log(`%cPlayer skill threshold reached (Level ${this.skillLevel + 1})! Ready to spawn bot.`, "color: green; font-weight: bold;");
                this.readyToSpawnBot = true;
                // Don't reset counter here, let cloneForBot handle it
            }
            return true;
        }
        return false;
    }

     shouldSpawnBot() {
         // Just returns the flag set by checkBotSpawnCondition
         return this.readyToSpawnBot;
     }


    cloneForBot() {
        if (!this.isInitialized || !this.readyToSpawnBot) {
            console.warn("Cannot clone for bot: Hierarchy not initialized or spawn condition not met.");
            return null;
        }

        console.log(`Spawning Bot for Skill Level ${this.skillLevel + 1}...`);

        // Create a new config, copying relevant parameters
        const botConfig = {
            numLevels: this.numLevels,
            learningRate: this.learningRate, // Bots might have different learning rates later
            streamNames: [...this.streamNames], // Copy stream names
            layerLatentSizes: [...this.layerLatentSizes], // Copy layer sizes
            anomalyThreshold: this.anomalyThreshold,
            anomalyHistorySize: this.anomalyHistorySize,
            botSpawnPatience: this.botSpawnPatience // Bot doesn't spawn bots itself usually
        };

        const botHierarchy = new CorticalHierarchy(botConfig);
        botHierarchy.defineStreams(this.streamNames); // Define streams for the bot

        // Initialize models structure (important!)
        // Need to wait for this before setting weights
         return botHierarchy.initializeModels().then(() => {
            // Deep copy weights from player models to bot models
            console.log("Cloning model weights...");
            for (let i = 0; i < this.numLevels; i++) {
                const playerLayer = this.layers[i];
                const botLayer = botHierarchy.layers[i];

                botLayer.forwardModel.setWeights(
                    playerLayer.forwardModel.getWeights().map(w => w.clone())
                );
                botLayer.backwardModel.setWeights(
                    playerLayer.backwardModel.getWeights().map(w => w.clone())
                );
                 // Copy optimizer state? Maybe not, let the bot start fresh or fine-tune? Start fresh.
            }
            console.log("Bot cloned successfully.");

            // Reset player's spawn condition and increment skill level
            this.readyToSpawnBot = false;
            this.botSpawnCounter = 0;
            this.skillLevel++;
            // Optionally reset anomaly history for the player to start tracking for the *next* level
            // this.anomalyHistory = Array(this.numLevels).fill(0).map(() => []);
            // this.averageAnomalies = Array(this.numLevels).fill(Infinity);
            console.log(`Player skill level advanced to: ${this.skillLevel}. Spawn counter reset.`);


            return botHierarchy; // Return the fully initialized and weighted bot hierarchy
         });
    }

    // Utility to get weights (e.g., for saving)
    getWeights() {
        if (!this.isInitialized) return null;
        return this.layers.map(layer => ({
            forward: layer.forwardModel.getWeights().map(w => w.arraySync()), // Convert to serializable format
            backward: layer.backwardModel.getWeights().map(w => w.arraySync())
        }));
    }

    // Utility to set weights (e.g., for loading)
    async setWeights(weightsData) {
        if (!this.isInitialized || !weightsData || weightsData.length !== this.numLevels) {
            console.error("Cannot set weights: Hierarchy not initialized or data mismatch.");
            return;
        }
        console.log("Setting weights...");
        for (let i = 0; i < this.numLevels; i++) {
            const layer = this.layers[i];
            const data = weightsData[i];
            // Ensure models are built (happens in initializeModels) before setting weights
            layer.forwardModel.setWeights(data.forward.map(wArray => tf.tensor(wArray)));
            layer.backwardModel.setWeights(data.backward.map(wArray => tf.tensor(wArray)));
        }
        console.log("Weights set successfully.");
    }
}
