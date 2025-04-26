 # Abyssal Arena: Echoes of Descent - Project README

 ## Introduction

 Welcome to the Abyssal Arena! This project aims to create a cooperative/competitive MMORPG inspired by the themes of *Made in Abyss*. The core concept revolves around a dynamic arena representing the Abyss, where player actions directly shape the environment and the AI opponents they face.

 This game leverages two key custom technologies:

 1.  **Cortical Library:** A JavaScript library (using tfjs/danfojs concepts internally, but self-contained) that processes player actions and perceived game state changes through a hierarchical, predictive model. It calculates 'Anomaly' scores based on prediction errors and can spawn 'Echo' bots that mimic learned player patterns.
 2.  **ToolArtifact (LLM Tool):** A specialized Large Language Model tool designed to generate other in-game tools, items, and potentially narrative elements ('Artifacts') based on complex contexts derived from gameplay and the Cortical Library's analysis.

 We will follow an iterative MVP (Minimum Viable Product) approach, ensuring we have a playable and demonstrable increment at each stage, much like building a skateboard before aiming for a car.

 ## Core Concepts

 *   **The Arena:** A persistent world structured as concentric Rings, representing descending layers of the Abyss. Deeper rings mean higher intensity gameplay.
 *   **Cortical Library:**
     *   **Unified Action/Sense Processing:** Treats player inputs and perceived game state changes as data streams.
     *   **Prediction & Anomaly:** Constantly predicts the next state; the difference between prediction and reality is the Anomaly score. High Anomaly = "Curse", Low Anomaly = potential "Blessing".
     *   **Echo Bots:** Clones derived from learned player patterns, populating the Arena as adaptive AI opponents.
 *   **ToolArtifact:** An LLM accessed via an API that can generate game content (items, tools, minor events) based on gameplay triggers, Cortical Library outputs, and potentially player intent signals (including EEG in later stages).
 *   **Adaptation:** A character progression system tied to surviving and thriving in deeper Rings, unlocking access and resistances.
 *   **Ascent Curses:** Penalties incurred when moving from deeper to shallower Rings.
 *   **Deep Bloom Event:** A periodic server-wide event increasing difficulty and potentially leveraging scaled LLM resources for major Artifact creation.

 ## Technology Stack (Initial Plan - Flexible for Jam)

 *   **Game Engine:** Phaser.js (for 2D web-based focus) or Three.js/Babylon.js (if aiming for 2.5D/3D later). Choose based on team expertise and MVP1 goals.
 *   **Cortical Library:** Custom Vanilla JS library (potentially using tfjs/danfojs for internal tensor/dataframe operations if needed, but should ideally be self-contained for ease of use).
 *   **ToolArtifact:** Requires API access to a capable LLM (e.g., via OpenAI API, Google AI API, local model endpoint) and the ToolArtifact wrapper/interface.
 *   **Server:** Node.js with WebSockets (e.g., Socket.IO) for multiplayer communication.
 *   **Database:** Simple DB (like SQLite, MongoDB) for player state persistence.
 *   **(Optional Later):** Web EEG libraries (e.g., BrainFlow WebSockets, MuseJS) for EEG integration.

 ## MVP Roadmap

 ---

 ### **MVP 1: The Skateboard - Core Loop Proof of Concept**

 *   **Goal:** Demonstrate the absolute basic loop: Player exists, acts, and the system *records* it. Test basic Cortical Library data ingestion.
 *   **Core Features:**
     *   Single Player only.
     *   One Arena "Ring" (a simple playable area).
     *   Player character can move (keyboard/mouse/touch).
     *   Basic interactable objects (e.g., collectible items, simple obstacles).
     *   Minimal UI showing basic player status.
 *   **Cortical Library Functionality:**
     *   **Data Collection:** Successfully capture player actions (movement inputs, interaction attempts) and simple senses (player position, object proximity/interaction result) into a structured format (e.g., timestamped JSON objects or simple arrays).
     *   **No Prediction/Anomaly Yet:** Focus solely on reliable data capture from the game state into the library's input buffer.
 *   **ToolArtifact/LLM Functionality:** None.
 *   **Playable Outcome:** A player can move around a simple space and interact with objects. Developers can verify that the Cortical Library is correctly receiving action/sense data streams. *Value: Proves basic engine setup and data pipeline.*

 ---

 ### **MVP 2: The Scooter - Basic Prediction & Feedback**

 *   **Goal:** Introduce the concept of prediction and anomaly feedback. Add a simple antagonist.
 *   **Core Features:**
     *   Simple, non-adaptive enemy type spawns.
     *   Basic combat or interaction mechanic with enemies (e.g., click to attack/push).
     *   UI element displaying a rudimentary "Anomaly Score".
     *   Visual/Audio feedback when Anomaly score changes significantly.
     *   Introduction of a boundary representing the edge of the first Ring.
 *   **Cortical Library Functionality:**
     *   **Basic Prediction:** Implement a simple predictive model (e.g., predicting the *next likely* player position based on recent velocity, or predicting success/failure of a basic action).
     *   **Anomaly Calculation:** Calculate a simple anomaly score (e.g., Mean Squared Error between predicted and actual position/state).
     *   **Data Storage:** Persist collected action/sense data locally for analysis.
 *   **ToolArtifact/LLM Functionality:** None.
 *   **Playable Outcome:** Player can move, interact with objects, and engage simple enemies. They see an Anomaly score react to their actions (e.g., moving erratically increases it, predictable movement decreases it). *Value: Demonstrates the core feedback loop of action -> prediction -> anomaly.*

 ---

 ### **MVP 3: The Bicycle - Echoes and Basic Adaptation**

 *   **Goal:** Introduce the first adaptive AI (Echoes) and basic multiplayer/progression concepts.
 *   **Core Features:**
     *   Basic Networking: Players can see each other move in the Arena (ghosts). No direct interaction yet.
     *   Second Arena Ring added with slightly tougher basic enemies.
     *   Simple "Adaptation" score tracked, increases slowly while in Ring 1. Ring 2 access requires Adaptation > threshold.
     *   UI displays Adaptation score.
 *   **Cortical Library Functionality:**
     *   **Simple Echo Spawning:** Ability to save a snapshot of learned patterns (the simple prediction model weights/data) from a player session. Load these patterns into a basic enemy AI ("Echo") that tries to replicate the *movements* or *simple action sequences* it learned. Echoes replace some basic enemies.
     *   **Refined Anomaly:** Anomaly score calculation is slightly more sophisticated, potentially influencing simple visual effects (minor screen tint for high anomaly - Curse hint, slight glow for low anomaly - Blessing hint).
     *   **Pattern Storage:** Server-side storage of anonymized player patterns for generating Echoes.
 *   **ToolArtifact/LLM Functionality:** None.
 *   **Playable Outcome:** Players can explore two rings (if adapted), see other players as ghosts, and encounter simple "Echo" bots that crudely mimic movement patterns observed from players. Anomaly has subtle visual effects. *Value: Introduces adaptive AI core and basic progression/multiplayer foundation.*

 ---

 ### **MVP 4: The Motorcycle - Crafting, Curses, and Co-op/PvP Hints**

 *   **Goal:** Implement core game systems (Curses, full Adaptation), introduce basic LLM tool usage, and enable rudimentary player interaction.
 *   **Core Features:**
     *   Multiple Rings (3-4) with increasing difficulty and specific environmental hazards.
     *   Full Adaptation system implemented: Requires specific levels to enter deeper rings, grants minor passive resistances. Gained via activity in deeper rings.
     *   Ascent Curses implemented: Trigger specific debuffs (visual distortions, stat reduction) when moving from deeper to shallower rings.
     *   Basic Player Interaction Zones: Designated areas where players can optionally duel (PvP) or must cooperate to overcome a challenge (PvE).
     *   Basic Inventory system.
 *   **Cortical Library Functionality:**
     *   **Adaptive Echoes:** Echoes use more sophisticated learned patterns (e.g., simple combat sequences, preferred ranges). Their difficulty implicitly scales with the patterns learned in the Ring they spawn in.
     *   **Anomaly -> Curse/Blessing v1:** High Anomaly actively triggers Ascent Curse-like debuffs *within* a ring. Sustained Low Anomaly grants minor temporary buffs (e.g., faster movement).
 *   **ToolArtifact/LLM Functionality:**
     *   **Basic Artifact Generation:** Trigger ToolArtifact API call on specific events (e.g., surviving a major Curse event, defeating a powerful Echo). Pass basic context (player ID, event type, Ring level, recent Anomaly score).
     *   **Simple Tool Output:** LLM generates a description of a *simple consumable* or temporary buff item (e.g., "Anomaly Dampener," "Survivor's Speed Boost"). Game logic grants this item to the player.
 *   **Playable Outcome:** A challenging multi-ring experience where players manage Adaptation and Anomaly, face adaptive Echoes, experience Ascent Curses, can optionally interact with others, and can earn simple generated items via the first integration of ToolArtifact. *Value: Delivers the core MMORPG loop with adaptive AI and initial procedural content generation.*

 ---

 ### **MVP 5: The Car - Deep Integration and Advanced Features**

 *   **Goal:** Integrate deeper LLM capabilities, introduce complex artifacts, potentially add EEG layer, and implement major events.
 *   **Core Features:**
     *   Deepest Rings (5-6) with unique, extreme challenges and high Adaptation requirements.
     *   Complex NPCs (e.g., an AI-driven Reg character) interacting with the world and players, potentially using LLM for behavior/dialogue informed by the Cortical Library.
     *   White Whistle / Advanced Artifact system: Generated via ToolArtifact using rich context (player history, anomaly patterns, potentially EEG data, specific sacrificial triggers), granting unique, powerful abilities.
     *   Deep Bloom Event implemented: Periodic server-wide challenge with scaled difficulty, LLM resource scaling, and unique rewards.
     *   (Optional) EEG Integration: If hardware/APIs are available, allow connection. Clean EEG signals subtly influence Anomaly calculation, potentially unlock specific neurofeedback mechanics in deep rings, and provide richer context for LLM-generated artifacts/responses.
     *   Full MMORPG features: Guilds, enhanced trading, leaderboards.
 *   **Cortical Library Functionality:**
     *   **Hierarchical Complexity:** Library simulates increased "cortical depth" for deeper rings, allowing learning of more complex patterns.
     *   **Rich Context Output:** Provides detailed summarized data (patterns, anomaly history, key events) to the LLM for ToolArtifact and NPC driving.
     *   **EEG Stream Processing (Optional):** Ingests and performs basic validation/processing of EEG data streams, passing relevant features (coherence, band power) to the LLM module.
 *   **ToolArtifact/LLM Functionality:**
     *   **Complex Generation:** Generates intricate Artifacts with unique mechanics based on rich context from Library/EEG.
     *   **Narrative/NPC Logic:** Potentially drives NPC dialogue, goals, and reactions based on world state and player interactions.
     *   **Deep Bloom Scaling:** Leverages potentially scaled cloud LLM resources during the event for mass analysis and artifact generation.
 *   **Playable Outcome:** A feature-rich MMORPG experience with deep AI adaptation, meaningful consequences for actions (Curses/Blessing/Transformation), unique procedurally generated high-tier items via LLM, major server events, and an optional cutting-edge EEG integration layer. *Value: Achieves the full vision of a dynamically evolving world shaped by player behavior and advanced AI.*

 ---

 ## Getting Started (Game Jam / Initial Setup)

 1.  **Clone the Repository:**
```bash
git clone [repository-url]
cd abyssal-arena
```
 2.  **Install Dependencies:**
     *   Install Node.js LTS.
     *   Install game engine dependencies (e.g., `npm install` for Phaser/Node.js projects).
     *   Set up API keys for ToolArtifact/LLM access (create a `.env` file based on `.env.example`).
 3.  **Run the Project (Example for Node.js/Phaser):**
```bash
npm install
npm start
```
 4.  **Focus on MVP 1:** Check the `mvp1` branch (or relevant tag/folder) and start building the core features outlined above! Coordinate tasks via GitHub Issues.

 ## Contribution Guidelines

 *   Please follow the coding style guides (to be defined).
 *   Create feature branches for your work (e.g., `feature/mvp1-player-movement`).
 *   Submit Pull Requests for review before merging to main/development branches.
 *   Document your code clearly.
 *   Communicate frequently, especially during a game jam!

 ## Future Work (Post-MVP 5)

 *   Advanced EEG Inter-brain coherence features.
 *   More sophisticated LLM-driven narrative systems.
 *   Performance optimization and scaling for large player counts.
 *   Expanded world content and lore integration.
