// stch.js - SpatioTemporal Cortical Hierarchy Library
// Uses TensorFlow.js (tfjs)

class STCHLevel {
    constructor(levelIndex, inputDim, encodingDim, learningRate, tf) {
        this.tf = tf;
        this.levelIndex = levelIndex;
        this.inputDim = inputDim;
        this.encodingDim = encodingDim;
        this.learningRate = learningRate;
        this.model = this._buildModel();
        this.isTraining = false;
        this.lastAnomaly = 0;
        console.log(`Level ${levelIndex}: Input ${inputDim}, Encoding ${encodingDim}`);
    }

    _buildModel() {
        const model = this.tf.sequential();
        // Encoder part
        model.add(this.tf.layers.dense({
            units: Math.max(this.encodingDim, Math.floor((this.inputDim + this.encodingDim) / 2)), // Hidden layer
            inputShape: [this.inputDim],
            activation: 'relu',
            kernelInitializer: 'randomNormal', // Explicitly initialize weights
            biasInitializer: 'zeros'
        }));
        model.add(this.tf.layers.dense({ // Encoding layer
            units: this.encodingDim,
            activation: 'tanh', // Use tanh for encoded representation (-1 to 1)
            name: `encoder_${this.levelIndex}`,
             kernelInitializer: 'randomNormal',
             biasInitializer: 'zeros'
        }));
        // Decoder part
        model.add(this.tf.layers.dense({
            units: Math.max(this.encodingDim, Math.floor((this.inputDim + this.encodingDim) / 2)), // Hidden layer mirror
            activation: 'relu',
             kernelInitializer: 'randomNormal',
             biasInitializer: 'zeros'
        }));
        model.add(this.tf.layers.dense({ // Output layer (predicting input)
            units: this.inputDim,
            activation: 'linear', // Linear activation for reconstruction
             kernelInitializer: 'randomNormal',
             biasInitializer: 'zeros'
        }));

        model.compile({
            optimizer: this.tf.train.adam(this.learningRate),
            loss: 'meanSquaredError' // MSE for reconstruction/prediction error
        });
        return model;
    }

    // Gets the output of the encoding layer
    encode(inputTensor) {
         // Find the encoder layer by name
         const encoderLayer = this.model.getLayer(`encoder_${this.levelIndex}`);
         if (!encoderLayer) {
             console.error(`Encoder layer not found for level ${this.levelIndex}`);
             // Fallback or default tensor: return a tensor of zeros with the expected shape
             return this.tf.zeros([inputTensor.shape[0], this.encodingDim]);
         }
        // Create a temporary model to get intermediate output
        const encoderModel = this.tf.model({
            inputs: this.model.inputs,
            outputs: encoderLayer.output
        });
        const encoding = encoderModel.predict(inputTensor);
        // Ensure the temporary model's resources are disposed if not managed by tf.tidy
        // Note: predict inside tf.tidy should handle intermediate tensors, but explicit disposal of the temporary model itself might be needed if it were stateful or larger.
        // Since it's just defining structure, tf.tidy around the call should suffice.
        // encoderModel.dispose(); // Generally not needed if predict is within tidy.
        return encoding;
    }

    // Predicts the input for this level based on the encoding from the level above
    // Note: For STCH, the *decoder* part of the *current* level model is used.
    // The input to this function should conceptually be the *prediction* from the level *above*.
    // However, the standard autoencoder predicts its own input.
    // For predictive coding: We need a way to predict Level(i-1)'s state from Level(i)'s state.
    // Let's adjust: The model predicts its own input. Anomaly is calculated based on this prediction vs actual input.
    // The *encoding* is passed upwards. The *prediction* of the *next* state comes from the top-down pass.
    predict(inputTensor) {
        return this.model.predict(inputTensor); // Autoencoder predicts its input
    }

    // Predicts the input for the *lower* level (Level i-1) given the encoding from *this* level (Level i)
    // This requires only the decoder part of the model.
    decode(encodingTensor) {
        // Find the encoder layer index to split the model
        let encoderLayerIndex = -1;
        for(let i = 0; i < this.model.layers.length; i++) {
            if(this.model.getLayer(undefined, i).name.startsWith('encoder_')) {
                encoderLayerIndex = i;
                break;
            }
        }

        if (encoderLayerIndex === -1) {
            console.error(`Decoder start not found for level ${this.levelIndex}`);
            return this.tf.zeros([encodingTensor.shape[0], this.inputDim]); // Return zero prediction
        }

        // Create a temporary model representing only the decoder part
        const decoderInput = this.tf.input({shape: [this.encodingDim]});
        let decoderOutput = decoderInput;
        for (let i = encoderLayerIndex + 1; i < this.model.layers.length; i++) {
            decoderOutput = this.model.getLayer(undefined, i).apply(decoderOutput);
        }
        const decoderModel = this.tf.model({ inputs: decoderInput, outputs: decoderOutput });

        const prediction = decoderModel.predict(encodingTensor);
        // decoderModel.dispose(); // Dispose temporary model structure - check tfjs docs if needed.
        return prediction;
    }


    async train(inputTensor) {
        if (this.isTraining) {
            // console.log(`Level ${this.levelIndex} skipping training: already in progress.`);
            return;
        }
        this.isTraining = true;
        // console.log(`Level ${this.levelIndex} starting training...`);

        try {
            const history = await this.model.fit(inputTensor, inputTensor, {
                epochs: 1, // Train for one epoch per trigger
                batchSize: 1, // Process the single current state
                verbose: 0 // Suppress console logging from tfjs fit
            });
            // console.log(`Level ${this.levelIndex} training complete. Loss: ${history.history.loss[0]}`);
            this.lastAnomaly = history.history.loss[0]; // Update last anomaly with training loss
        } catch (error) {
            console.error(`Error training level ${this.levelIndex}:`, error);
             // Reset flag even if error occurs
             this.isTraining = false;
             // Consider how to handle the error, e.g., retry later, log persistently
             throw error; // Re-throw if the caller needs to know
        } finally {
             // Ensure the flag is reset whether training succeeded or failed
             this.isTraining = false;
             // console.log(`Level ${this.levelIndex} training flag reset.`);
        }
    }

    getWeights() {
        return this.model.getWeights().map(w => w.clone());
    }

    setWeights(weights) {
        this.model.setWeights(weights.map(w => w.clone())); // Clone to avoid sharing tensors
    }

    dispose() {
        this.model.dispose();
    }
}

class STCHierarchy {
    constructor(config = {}, tf) {
         if (!tf) {
            throw new Error("TensorFlow.js instance (tf) must be provided to STCHierarchy.");
        }
        this.tf = tf;
        this.config = {
            streams: {}, // { streamName: dimension, ... }
            hierarchyParams: [ // Defines structure: [level0_encoding_dim, level1_encoding_dim, ...]
                { encodingDim: 64, anomalyThreshold: 0.1 },
                { encodingDim: 32, anomalyThreshold: 0.05 }
            ],
            learningRate: 0.001,
            anomalyDecay: 0.95, // Factor to decay anomaly thresholds over time (stability)
            stabilityThreshold: 0.01, // Average anomaly level required for stability
            stabilityWindow: 100, // Number of steps to average anomaly over for stability check
            botSpawnCooldown: 200, // Minimum steps between bot spawns
            ...config
        };

        this.streams = {}; // { streamName: { dimension: dim, startIndex: idx } }
        this.streamOrder = []; // Keep track of concatenation order
        this.totalInputDim = 0;
        this.levels = []; // Array of STCHLevel instances
        this.levelAnomalies = []; // Store current anomaly per level
        this.recentAnomalies = []; // Store history for stability check
        this.stepsSinceLastSpawn = 0;
        this.isBuilt = false;

        this._initializeStreams(this.config.streams);
        if (Object.keys(this.streams).length > 0) {
            this._buildHierarchy();
        }
    }

    _initializeStreams(streamConfig) {
        this.streams = {};
        this.streamOrder = [];
        this.totalInputDim = 0;
        Object.entries(streamConfig).forEach(([name, dimension]) => {
            this.addStream(name, dimension, false); // Add streams without rebuilding each time
        });
        console.log("Initialized streams:", this.streams);
        console.log("Total input dimension:", this.totalInputDim);
    }

    _buildHierarchy() {
        // Dispose existing levels if rebuilding
        this.levels.forEach(level => level.dispose());
        this.levels = [];
        this.levelAnomalies = [];

        let currentInputDim = this.totalInputDim;
        if (currentInputDim <= 0) {
            console.warn("Cannot build hierarchy: No streams defined or total input dimension is zero.");
            this.isBuilt = false;
            return;
        }

        this.config.hierarchyParams.forEach((params, i) => {
            const level = new STCHLevel(
                i,
                currentInputDim,
                params.encodingDim,
                this.config.learningRate,
                this.tf
            );
            this.levels.push(level);
            this.levelAnomalies.push(0); // Initialize anomaly tracking
            currentInputDim = params.encodingDim; // Input for next level is encoding dim of current
        });
        this.isBuilt = true;
        console.log(`Hierarchy built with ${this.levels.length} levels.`);
    }

    addStream(streamName, dimension, rebuild = true) {
        if (this.streams[streamName]) {
            console.warn(`Stream ${streamName} already exists. Updating dimension.`);
            // Adjust total dimension (more complex if in middle, easier if just updating)
             if (this.streams[streamName].dimension !== dimension) {
                 console.warn(`Dimension change for ${streamName} requires careful handling if hierarchy already built. Rebuilding.`);
                 rebuild = true; // Force rebuild on dimension change
                 // Simple approach: remove and re-add logic
                 const oldDim = this.streams[streamName].dimension;
                 this.totalInputDim -= oldDim;
                 delete this.streams[streamName];
                 const index = this.streamOrder.indexOf(streamName);
                 if (index > -1) this.streamOrder.splice(index, 1);
                 // Now add it back
             } else {
                 return; // No change needed
             }
        }

        const startIndex = this.totalInputDim;
        this.streams[streamName] = { dimension, startIndex };
        this.streamOrder.push(streamName); // Maintain order
        this.totalInputDim += dimension;

        console.log(`Added stream: ${streamName} (dim: ${dimension}). Total dim: ${this.totalInputDim}`);


        if (rebuild && this.totalInputDim > 0) {
            console.log("Rebuilding hierarchy due to stream addition/change.");
            this._buildHierarchy();
        } else if (!rebuild && !this.isBuilt && this.totalInputDim > 0) {
             // If adding streams initially without rebuild flag, build once all are added.
             // This check might be redundant depending on how initialization is called.
             // console.log("Building hierarchy after initial stream additions.");
             // this._buildHierarchy();
        } else if (this.totalInputDim <= 0) {
            console.warn("Total input dimension is still zero after adding stream. Hierarchy not built.");
            this.isBuilt = false;
        }
    }

     removeStream(streamName, rebuild = true) {
        if (!this.streams[streamName]) {
            console.warn(`Stream ${streamName} does not exist.`);
            return;
        }

        const { dimension } = this.streams[streamName];
        this.totalInputDim -= dimension;
        delete this.streams[streamName];

        const index = this.streamOrder.indexOf(streamName);
        if (index > -1) {
            this.streamOrder.splice(index, 1);
        }

        // Recalculate start indices for remaining streams
        let currentStartIndex = 0;
        this.streamOrder.forEach(name => {
            this.streams[name].startIndex = currentStartIndex;
            currentStartIndex += this.streams[name].dimension;
        });

         console.log(`Removed stream: ${streamName}. Total dim: ${this.totalInputDim}`);

        if (rebuild && this.totalInputDim > 0) {
             console.log("Rebuilding hierarchy due to stream removal.");
            this._buildHierarchy();
        } else if (this.totalInputDim <= 0) {
            console.warn("Total input dimension is zero after removing stream. Disposing hierarchy.");
            this.levels.forEach(level => level.dispose());
            this.levels = [];
            this.levelAnomalies = [];
            this.isBuilt = false;
        }
    }

    _concatenateInputs(inputData) {
        const orderedInputs = [];
        for (const streamName of this.streamOrder) {
            const data = inputData[streamName];
            const expectedDim = this.streams[streamName]?.dimension;

            if (!data) {
                 console.warn(`Input data missing for stream: ${streamName}. Using zeros.`);
                 if (expectedDim === undefined) {
                      console.error(`Stream ${streamName} configuration missing. Cannot determine dimension.`);
                      // Handle error appropriately, maybe throw or return null tensor
                      return null;
                 }
                 orderedInputs.push(...Array(expectedDim).fill(0));
                 continue; // Skip to next stream
             }

             if (expectedDim === undefined) {
                 console.error(`Stream ${streamName} configuration missing, but data provided. Skipping.`);
                 continue;
             }


            if (data.length !== expectedDim) {
                console.warn(`Input data for stream ${streamName} has incorrect dimension. Expected ${expectedDim}, got ${data.length}. Padding/truncating.`);
                // Basic padding/truncating
                const adjustedData = Array(expectedDim).fill(0);
                for(let i=0; i< Math.min(data.length, expectedDim); i++){
                    adjustedData[i] = data[i];
                }
                 orderedInputs.push(...adjustedData);
            } else {
                 orderedInputs.push(...data);
            }
        }
        // Ensure the total length matches expected totalInputDim
         if (orderedInputs.length !== this.totalInputDim) {
             console.error(`Concatenated input length ${orderedInputs.length} does not match total expected dimension ${this.totalInputDim}. Stream config mismatch?`);
              // Handle this error robustly - maybe return null or throw
             // For now, attempt to create tensor but it might fail or be incorrect
             // return null; // Safer option
             // Pad with zeros if short, truncate if long? Risky.
             const finalInput = Array(this.totalInputDim).fill(0);
             for(let i=0; i< Math.min(orderedInputs.length, this.totalInputDim); i++){
                 finalInput[i] = orderedInputs[i];
             }
              return this.tf.tensor2d([finalInput], [1, this.totalInputDim]); // Create tensor from corrected array
         }

        return this.tf.tensor2d([orderedInputs], [1, this.totalInputDim]); // Shape [1, totalInputDim]
    }

     _deconcatenateOutput(flatOutputTensor) {
        const outputData = {};
        const flatArray = flatOutputTensor.dataSync(); // Get data as flat array

        for (const streamName of this.streamOrder) {
            const streamInfo = this.streams[streamName];
            if (streamInfo) {
                 const { dimension, startIndex } = streamInfo;
                 // Ensure slice indices are within bounds
                 const endIndex = startIndex + dimension;
                 if (startIndex < flatArray.length && endIndex <= flatArray.length) {
                    outputData[streamName] = Array.from(flatArray.slice(startIndex, endIndex));
                 } else {
                     console.warn(`Cannot extract stream ${streamName}: Indices [${startIndex}, ${endIndex}) out of bounds for flat array length ${flatArray.length}. Returning zeros.`);
                     outputData[streamName] = Array(dimension).fill(0);
                 }
            }
        }
        return outputData;
    }


   async processStep(inputData) {
    if (!this.isBuilt || this.levels.length === 0) {
        // console.warn("Hierarchy not built or no levels exist. Cannot process step.");
        return { anomalies: [], prediction: {}, botSpawned: null };
    }

    this.stepsSinceLastSpawn++;
    let botSpawned = null; // Initialize botSpawned for this step

    // Wrap the core processing in tf.tidy to manage intermediate tensors
    return this.tf.tidy(() => {
        const flatInputTensor = this._concatenateInputs(inputData);
         if (!flatInputTensor) {
             console.error("Failed to create flat input tensor. Aborting step.");
              return { anomalies: [], prediction: {}, botSpawned: null }; // Return default/empty result
         }


        // --- Forward Pass (Encoding) ---
        const levelEncodings = [];
        let currentLevelInput = flatInputTensor;
        for (let i = 0; i < this.levels.length; i++) {
            const encoding = this.levels[i].encode(currentLevelInput);
            levelEncodings.push(encoding);
            currentLevelInput = encoding; // Output of this level is input to the next
        }

        // --- Backward Pass (Prediction) ---
        const levelPredictions = new Array(this.levels.length);
        let nextLevelPrediction = levelEncodings[this.levels.length - 1]; // Start with highest encoding

        for (let i = this.levels.length - 1; i >= 0; i--) {
             // The decoder at level 'i' predicts the input *for* level 'i'
             // based on the representation from level 'i+1' (or the top encoding).
             // So, we use the decoder of level 'i' with the input 'nextLevelPrediction'
             // which comes from the level above it.
            const predictionForLowerLevel = this.levels[i].decode(nextLevelPrediction);
            levelPredictions[i] = predictionForLowerLevel;

            // The input for the *next* iteration (level i-1's prediction)
            // conceptually should be based on the *output* of level i's decoder.
            // However, the *input* to level i's decoder was the encoding/prediction from level i+1.
            // Let's rethink: The prediction pass generates predictions P_i for each level i's input.
            // P_N-1 is generated from E_N-1 (highest encoding).
            // P_N-2 is generated from P_N-1 via Decoder_N-1. No, that's not right.
            // P_N-2 should be generated from E_N-1 (top encoding) via Decoder_N-1.
            // P_i should be generated from E_top via the chain of decoders down to level i+1.

            // Correction: Prediction should flow down using the hierarchy's state.
            // Let's use the standard autoencoder prediction for anomaly calculation first.
            // Then consider the top-down generative prediction separately.

            // --- Anomaly Calculation ---
             // Anomaly at level 'i' is the difference between the *actual* input to level 'i'
             // and the *reconstruction* of that input produced by the autoencoder at level 'i'.
             const actualInputForLevel = (i === 0) ? flatInputTensor : levelEncodings[i - 1];
             // Get the reconstruction from the *full* autoencoder model at level i
             const reconstruction = this.levels[i].predict(actualInputForLevel); // Full AE prediction

             // Calculate Mean Squared Error as anomaly score
             const anomalyTensor = this.tf.losses.meanSquaredError(actualInputForLevel, reconstruction);
             const anomaly = anomalyTensor.dataSync()[0]; // Get scalar value
             this.levelAnomalies[i] = anomaly;
             this.levels[i].lastAnomaly = anomaly; // Store on level object too

             // Trigger learning if anomaly exceeds threshold
             const threshold = this.config.hierarchyParams[i]?.anomalyThreshold ?? 0.1;
             if (anomaly > threshold && !this.levels[i].isTraining) {
                 // Don't await, let it run in background
                 this.levels[i].train(actualInputForLevel).catch(err => {
                      console.error(`Background training failed for level ${i}:`, err);
                      // Handle error, maybe retry logic or logging
                 });
             }

             // For the next iteration of the backward pass (predicting i-1),
             // we need the input to the decoder of level i-1. This should be E_i.
             // No, the *prediction* flows down.
             // Let's stick to the definition: Prediction P_i is generated by Decoder_i+1 from P_i+1.
             // And P_N-1 is generated by Decoder_N from E_N-1.
             // This seems right for predictive coding. Let's refine the 'decode' usage.

             // Re-attempt Backward Pass Logic for Predictive Coding:
             // Input to `decode` at level `i` should be the prediction `P_i+1` from level `i+1`.
             // Start: Input to highest decoder (Level N-1) is the highest encoding E_N-1.
             // Let `topDownPrediction = E_N-1` initially.
             // Loop i = N-1 down to 0:
             //   `predictionForLevel_i_Input = this.levels[i].decode(topDownPrediction)`
             //   `levelPredictions[i] = predictionForLevel_i_Input` // Store prediction of Level i's input
             //   `topDownPrediction = predictionForLevel_i_Input` // This prediction becomes input for decoder below
             // This seems incorrect. `decode(E_i)` should predict `Input_i`.

             // Let's reset the backward pass logic:
             // We need *one* final prediction of the *next* raw input state.
             // This is generated by the full top-down pathway.
             // Input: Highest encoding `E_N-1 = levelEncodings[this.levels.length - 1]`
             // Let `predictedRepresentation = E_N-1`
             // Loop i = N-1 down to 0:
             //   `predictedRepresentation = this.levels[i].decode(predictedRepresentation)`
             // The final `predictedRepresentation` after the loop is the `predictedFlatOutputTensor`.

             // We only need to run the full decode chain once after the forward pass.
        } // End initial loop (used mainly for anomaly calc and training trigger now)


        // --- Generate Final Top-Down Prediction ---
         let predictedNextFlatInput;
         if (levelEncodings.length > 0) {
            let topDownPrediction = levelEncodings[this.levels.length - 1]; // Start with highest encoding
            for (let i = this.levels.length - 1; i >= 0; i--) {
                topDownPrediction = this.levels[i].decode(topDownPrediction);
            }
            predictedNextFlatInput = topDownPrediction;
         } else {
            // Handle case with no levels - predict zeros or current input?
            predictedNextFlatInput = this.tf.zerosLike(flatInputTensor);
         }


        // --- Bot Spawning Logic ---
        const overallAnomaly = this.levelAnomalies.reduce((a, b) => a + b, 0) / this.levels.length;
        this.recentAnomalies.push(overallAnomaly);
        if (this.recentAnomalies.length > this.config.stabilityWindow) {
            this.recentAnomalies.shift(); // Keep window size
        }

        const avgAnomaly = this.recentAnomalies.reduce((a, b) => a + b, 0) / this.recentAnomalies.length;

        if (this.recentAnomalies.length === this.config.stabilityWindow &&
            avgAnomaly < this.config.stabilityThreshold &&
            this.stepsSinceLastSpawn > this.config.botSpawnCooldown)
        {
            console.log(`Stability achieved (Avg Anomaly: ${avgAnomaly.toFixed(4)}). Spawning Bot.`);
            botSpawned = this.clone(); // Clone current state for the bot
            this.recentAnomalies = []; // Reset stability check
            this.stepsSinceLastSpawn = 0;
            // Optional: Slightly increase anomaly thresholds after spawning to encourage new learning
            // this.config.hierarchyParams.forEach(p => p.anomalyThreshold *= 1.1);
        }

        // Deconcatenate the final prediction
        const predictedOutputData = this._deconcatenateOutput(predictedNextFlatInput);

        // Return results
        return {
            anomalies: [...this.levelAnomalies], // Return a copy
            prediction: predictedOutputData,
            botSpawned: botSpawned // Will be null or a new STCHierarchy clone
        };
      }); // End tf.tidy
    }


    clone() {
        // Create a new instance with the same config (deep copy config if mutable parts)
        const clonedConfig = JSON.parse(JSON.stringify(this.config));
        const clone = new STCHierarchy(clonedConfig, this.tf);

         // Ensure streams are setup correctly in the clone
         clone._initializeStreams(this.config.streams);
         if (!clone.isBuilt) {
             console.error("Failed to build hierarchy during cloning.");
             return null; // Or handle error appropriately
         }


        // Clone level weights
        if (this.levels.length !== clone.levels.length) {
             console.error("Cloning error: Level count mismatch between source and clone.", this.levels.length, clone.levels.length);
              // This indicates a potential issue in _buildHierarchy or config handling.
             return null;
         }
        for (let i = 0; i < this.levels.length; i++) {
            const weights = this.levels[i].getWeights();
             // Check if clone level exists and weights are valid before setting
             if (clone.levels[i] && weights && weights.length > 0) {
                 clone.levels[i].setWeights(weights);
                 // Dispose the copied weights from getWeights() after setting them in the clone
                 weights.forEach(w => w.dispose());
             } else {
                  console.error(`Cloning error: Cannot set weights for level ${i}. Clone level or weights invalid.`);
                  // Handle error: maybe stop cloning, return null, or log issue.
                  // weights?.forEach(w => w.dispose()); // Dispose weights even if not used
                  return null; // Indicate cloning failed
             }
        }

        // Clone other relevant state if necessary (e.g., anomaly history? maybe not)
        clone.stepsSinceLastSpawn = 0; // Reset for the new bot/clone

        return clone;
    }

     // Add methods for dynamic hierarchy adjustment (optional for V1)
     addHierarchyLevel(encodingDim = null, anomalyThreshold = null) {
         if (!this.isBuilt || this.levels.length === 0) {
             console.warn("Cannot add level: Hierarchy not built or is empty.");
             return;
         }

         const lastLevel = this.levels[this.levels.length - 1];
         const newLevelInputDim = lastLevel.encodingDim;
         const newEncodingDim = encodingDim ?? Math.max(1, Math.floor(newLevelInputDim / 2)); // Default heuristic
         const newThreshold = anomalyThreshold ?? this.config.hierarchyParams[this.levels.length -1]?.anomalyThreshold ?? 0.1; // Use last level's threshold or default

         console.log(`Adding new hierarchy level ${this.levels.length}: Input ${newLevelInputDim}, Encoding ${newEncodingDim}`);

         const newLevel = new STCHLevel(
             this.levels.length,
             newLevelInputDim,
             newEncodingDim,
             this.config.learningRate,
             this.tf
         );

         this.levels.push(newLevel);
         this.levelAnomalies.push(0);
         // Update config representation if needed
         this.config.hierarchyParams.push({ encodingDim: newEncodingDim, anomalyThreshold: newThreshold });
     }

     dispose() {
         console.log("Disposing STCHierarchy and all levels...");
         this.levels.forEach(level => level.dispose());
         this.levels = [];
         this.streams = {};
         this.streamOrder = [];
         this.totalInputDim = 0;
         this.isBuilt = false;
         console.log("STCHierarchy disposed.");
     }
}

// Bot class to wrap a cloned hierarchy
class Bot {
    constructor(id, hierarchy, actionStreamNames) {
        this.id = id;
        this.hierarchy = hierarchy; // The cloned STCHierarchy instance
        this.actionStreamNames = actionStreamNames; // Which streams represent its actions
        this.tf = hierarchy.tf; // Get tf instance from hierarchy
        this.currentAnomalies = [];
        this.skillLevel = 0; // Example skill metric
    }

    getAction(currentSenses) {
         if (!this.hierarchy || !this.hierarchy.isBuilt) {
             console.warn(`Bot ${this.id}: Hierarchy not available. Cannot get action.`);
             // Return default actions (e.g., zeros) based on expected action stream dimensions
             const defaultActions = {};
             this.actionStreamNames.forEach(name => {
                 const dim = this.hierarchy?.streams[name]?.dimension ?? 1; // Default dim 1 if unknown
                 defaultActions[name] = Array(dim).fill(0);
             });
             return defaultActions;
         }
        // The bot needs to predict its *next* actions based on current senses.
        // It runs its hierarchy with senses + *something* for its current actions.
        // What actions should it feed in? Perhaps the actions it took *last* step? Or zeros?
        // Let's assume it feeds in its *predicted* actions from the *previous* step as the 'current action' input.
        // This requires storing the last prediction. Or, maybe it just feeds in the senses and the hierarchy predicts the full next state including actions.

        // Simpler approach: Use the top-down prediction mechanism which predicts the *entire* next state vector.
        // We only need the *senses* to kick off the process if we assume the hierarchy implicitly predicts actions too.

        // Let's try running processStep with *only* senses and zeroed actions.
        // The prediction output should contain the predicted next actions.

        const botInput = { ...currentSenses }; // Start with senses
         // Add zero placeholders for action streams the bot controls
         this.actionStreamNames.forEach(name => {
            const streamInfo = this.hierarchy.streams[name];
            if (streamInfo) {
                botInput[name] = Array(streamInfo.dimension).fill(0);
             } else {
                 console.warn(`Bot ${this.id}: Action stream ${name} not found in hierarchy config.`);
             }
         });
          // Add zero placeholders for any *other* streams the hierarchy expects but weren't provided as senses
          this.hierarchy.streamOrder.forEach(name => {
              if (!botInput.hasOwnProperty(name)) {
                  const streamInfo = this.hierarchy.streams[name];
                   if (streamInfo) {
                       console.warn(`Bot ${this.id}: Input stream ${name} not provided as sense, using zeros.`);
                       botInput[name] = Array(streamInfo.dimension).fill(0);
                   }
              }
          });


         // Process this combined input to get the prediction
         // Use async/await here as processStep is async
         // Note: This runs the bot's *internal* simulation/prediction step.
         return this.hierarchy.processStep(botInput).then(results => {
             this.currentAnomalies = results.anomalies; // Store bot's own anomalies
             this.updateSkill();

             const predictedActions = {};
             this.actionStreamNames.forEach(name => {
                  if (results.prediction && results.prediction.hasOwnProperty(name)) {
                     predictedActions[name] = results.prediction[name];
                  } else {
                      // Fallback if prediction doesn't contain the action stream
                      const dim = this.hierarchy?.streams[name]?.dimension ?? 1;
                      predictedActions[name] = Array(dim).fill(0);
                      console.warn(`Bot ${this.id}: Predicted actions missing for stream ${name}. Using zeros.`);
                  }
             });
             return predictedActions; // Return only the predicted values for action streams
         }).catch(error => {
              console.error(`Bot ${this.id}: Error during getAction processing:`, error);
              // Return default actions on error
              const defaultActions = {};
              this.actionStreamNames.forEach(name => {
                  const dim = this.hierarchy?.streams[name]?.dimension ?? 1;
                  defaultActions[name] = Array(dim).fill(0);
              });
              return defaultActions;
         });


    }

     updateSkill() {
         // Example: Skill is inverse of average anomaly
         if (this.currentAnomalies && this.currentAnomalies.length > 0) {
             const avgAnomaly = this.currentAnomalies.reduce((a, b) => a + b, 0) / this.currentAnomalies.length;
             // Avoid division by zero, handle NaN
             if (!isNaN(avgAnomaly) && avgAnomaly > 1e-6) {
                 this.skillLevel = 1 / avgAnomaly;
             } else if (avgAnomaly <= 1e-6) {
                 this.skillLevel = 1e6; // High skill if anomaly is near zero
             } else {
                 this.skillLevel = 0; // Reset skill if anomaly is NaN or invalid
             }
         } else {
             this.skillLevel = 0; // No anomalies, no skill calculation yet
         }
     }

    dispose() {
        console.log(`Disposing Bot ${this.id} and its hierarchy...`);
        if (this.hierarchy) {
            this.hierarchy.dispose();
        }
         console.log(`Bot ${this.id} disposed.`);
    }
}

// Export classes if using modules, otherwise they are global in a single script
// export { STCHierarchy, Bot };
