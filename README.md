LSTM-Based Predator–Prey Evolution

This repository contains a C++/SFML project demonstrating a coevolutionary predator–prey simulation in which each agent (predator or prey) uses a Long Short-Term Memory (LSTM) neural network to control its behavior. Over multiple generations, a genetic algorithm (GA) evolves the LSTM parameters for both species, producing emergent hunting and evasion strategies.

--------------------------------------------------------------------------------
KEY FEATURES

1) Recurrent Neural Networks (LSTM)
   - Each agent’s decision process is governed by a single-layer LSTM (8 hidden units, 4 inputs, 2 outputs).
   - Agents can remember short-term context across timesteps, enabling more strategic or predictive behaviors.

2) Coevolution of Predator & Prey
   - Predators (red) must eat prey to gain energy; prey (green) must feed on yellow food items to survive.
   - Both species run the same type of LSTM architecture but evolve in separate populations.
   - An arms race dynamic emerges as predators adapt to be more efficient hunters, while prey learn better evasion.

3) Energy & Fitness Mechanics
   - Each agent loses energy over time.
   - Prey restore energy by eating food; predators restore energy by eating prey.
   - Agents gain fitness from survival time, plus bonus fitness for consuming food (prey) or prey (predators).

4) Genetic Algorithm
   - At the end of each generation (e.g., every 1500 steps), the best individuals reproduce via crossover and mutation.
   - The new generation starts in a fresh environment but retains the evolved LSTM parameters.

5) Real-Time 2D Visualization
   - Uses SFML to render in a 1200x800 window.
   - Red circles = predators, green circles = prey, yellow circles = food.

6) Data Logging
   - At the end of each generation, key statistics are appended to "stats.csv":
     * Generation index
     * Average & max fitness (predators and prey)
     * Number of prey eaten
   - Useful for analysis in Excel, Python, R, etc.

--------------------------------------------------------------------------------
PROJECT STRUCTURE

.
1. Constants.hpp         (Global simulation parameters)
2. Creature.hpp/.cpp     (Predator/Prey agent structure)
3. GA.hpp/.cpp           (Genetic Algorithm: crossover, mutation, evolvePopulation)
4. LSTMNetwork.hpp/.cpp  (Single-step LSTM forward pass)
5. Random.hpp/.cpp       (Global RNG, std::mt19937)
6. Simulation.hpp/.cpp   (Main simulation loop, environment updates, data logging)
7. main.cpp              (Entry point, runs the Simulation)
8. stats.csv             (Generated at runtime, logs generation data)

--------------------------------------------------------------------------------
BUILDING & RUNNING

Prerequisites:
- C++17 (or higher) compiler
- SFML installed (Graphics, Window, System)

Example Compilation (Linux/macOS with g++):
  g++ -std=c++17 main.cpp Simulation.cpp GA.cpp Creature.cpp LSTMNetwork.cpp Random.cpp \
      -lsfml-graphics -lsfml-window -lsfml-system \
      -o EvoLSTM
  ./EvoLSTM

On Windows, link SFML libraries in Visual Studio or MinGW, and place sfml-xxx.dll files next to your .exe if using dynamic linking.

--------------------------------------------------------------------------------
DATA LOGGING

- "stats.csv" is created (or appended) at runtime in the project folder.
- Columns:
    Gen, AvgPredFit, MaxPredFit, AvgPreyFit, MaxPreyFit, PreyEaten
- You can open stats.csv in Excel or any data analysis tool to plot fitness over time.

--------------------------------------------------------------------------------
POSSIBLE EXTENSIONS

1) Speciation/NEAT
   - Evolve network topologies dynamically and preserve genetic innovations.

2) Multi-Objective Optimization
   - Evaluate energy efficiency, survival, or other criteria simultaneously.

3) Advanced Resource / Obstacles
   - Add terrain or walls, more complex resource generation/regrowth.

4) Larger/Stacked LSTM
   - Increase hidden size or add layers for richer memory capacity.

5) Additional Species
   - Introduce multiple predator or prey types, each with unique parameters.

--------------------------------------------------------------------------------
CONTRIBUTING

- Issues: Please open an issue for bugs or suggestions.
- Pull Requests: Contributions that improve performance or add features are welcome.

--------------------------------------------------------------------------------
LICENSE

This project is distributed under the MIT License. See LICENSE for details.

--------------------------------------------------------------------------------
