#include "Simulation.hpp"
#include "Constants.hpp"
#include "GA.hpp"
#include "LSTMNetwork.hpp"
#include "Random.hpp"
#include <cmath>
#include <iostream>

static std::uniform_real_distribution<float> distX(0.f, (float)WINDOW_WIDTH);
static std::uniform_real_distribution<float> distY(0.f, (float)WINDOW_HEIGHT);

Simulation::Simulation()
    : window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "EvoSim LSTM Predator-Prey"),
    generation(1),
    steps(0),
    totalPreyEaten(0),
    wroteHeader(false)
{
    window.setFramerateLimit(FRAMERATE);
}

void Simulation::init()
{
    if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "Warning: Could not load 'arial.ttf'.\n";
    }
    genText.setFont(font);
    genText.setCharacterSize(20);
    genText.setFillColor(sf::Color::White);
    genText.setPosition(10.f, 10.f);

    // Create initial predators
    predators.clear();
    predators.reserve(NUM_PREDATORS);
    for (unsigned i = 0; i < NUM_PREDATORS; i++) {
        std::vector<float> genome(LSTM_GENOME_SIZE);
        for (auto& g : genome) {
            g = (dist01(rng) * 2.f - 1.f) * 0.5f;
        }
        Creature c(CreatureType::PREDATOR, distX(rng), distY(rng), genome);
        predators.push_back(c);
    }

    // Create initial prey
    prey.clear();
    prey.reserve(NUM_PREY);
    for (unsigned i = 0; i < NUM_PREY; i++) {
        std::vector<float> genome(LSTM_GENOME_SIZE);
        for (auto& g : genome) {
            g = (dist01(rng) * 2.f - 1.f) * 0.5f;
        }
        Creature c(CreatureType::PREY, distX(rng), distY(rng), genome);
        prey.push_back(c);
    }

    createFood(50);
}

void Simulation::createFood(unsigned count)
{
    food.clear();
    food.reserve(count);
    for (unsigned i = 0; i < count; i++) {
        Food f;
        f.x = distX(rng);
        f.y = distY(rng);
        f.eaten = false;
        food.push_back(f);
    }
}

void Simulation::run()
{
    init(); // setup generation 1

    while (window.isOpen()) {
        handleEvents();
        update();
        render();
    }
}

void Simulation::handleEvents()
{
    sf::Event ev;
    while (window.pollEvent(ev)) {
        if (ev.type == sf::Event::Closed) {
            window.close();
        }
    }
}

void Simulation::update()
{
    simulationStep();
    steps++;
    if (steps >= STEPS_PER_GENERATION) {
        // end of generation: log stats, evolve
        logStats(); // gather & write data to stats.csv
        nextGeneration();
        steps = 0;
        generation++;
        std::cout << "Generation " << generation << " begins.\n";
    }
}

void Simulation::render()
{
    window.clear(sf::Color(30, 30, 30));

    // Draw food
    for (auto& f : food) {
        if (!f.eaten) {
            sf::CircleShape shape(3.f);
            shape.setFillColor(sf::Color::Yellow);
            shape.setOrigin(3.f, 3.f);
            shape.setPosition(f.x, f.y);
            window.draw(shape);
        }
    }

    // Draw prey
    for (auto& py : prey) {
        if (!py.alive) continue;
        sf::CircleShape shape(5.f);
        shape.setFillColor(sf::Color::Green);
        shape.setOrigin(5.f, 5.f);
        shape.setPosition(py.x, py.y);
        window.draw(shape);
    }

    // Draw predators
    for (auto& pd : predators) {
        if (!pd.alive) continue;
        sf::CircleShape shape(7.f);
        shape.setFillColor(sf::Color::Red);
        shape.setOrigin(7.f, 7.f);
        shape.setPosition(pd.x, pd.y);
        window.draw(shape);
    }

    // Display generation
    genText.setString("Generation: " + std::to_string(generation));
    window.draw(genText);

    window.display();
}

void Simulation::nextGeneration()
{
    // Evolve both populations
    evolvePopulation(predators, CreatureType::PREDATOR, NUM_PREDATORS);
    evolvePopulation(prey, CreatureType::PREY, NUM_PREY);

    // Reset for next gen
    createFood(50);

    // Reset total prey eaten counter
    totalPreyEaten = 0;
}

void Simulation::simulationStep()
{
    // --- Predators ---
    for (auto& pd : predators) {
        if (!pd.alive) continue;

        // Find nearest prey
        float bestDist = 999999.f;
        float dx = 0.f, dy = 0.f;
        for (auto& pr : prey) {
            if (!pr.alive) continue;
            float dist = std::hypot(pd.x - pr.x, pd.y - pr.y);
            if (dist < bestDist) {
                bestDist = dist;
                dx = pr.x - pd.x;
                dy = pr.y - pd.y;
            }
        }
        if (bestDist > 900000.f) {
            bestDist = 0.f;
            dx = 0.f;
            dy = 0.f;
        }
        float len = std::hypot(dx, dy) + 0.0001f;
        dx /= len;
        dy /= len;

        // LSTM inputs
        float distFrac = bestDist / (float)std::max(WINDOW_WIDTH, WINDOW_HEIGHT);
        float energyFrac = pd.energy / 20.f;
        float inputArr[LSTM_INPUT_SIZE] = { distFrac, dx, dy, energyFrac };
        float outArr[LSTM_OUTPUT_SIZE];

        // Single-step LSTM
        lstmForward(pd.genome, inputArr, pd.hiddenState, pd.cellState, outArr);

        pd.vx = outArr[0] * MAX_SPEED;
        pd.vy = outArr[1] * MAX_SPEED;
        pd.x += pd.vx;
        pd.y += pd.vy;

        // Wrap
        if (pd.x < 0) pd.x += WINDOW_WIDTH;
        else if (pd.x >= WINDOW_WIDTH) pd.x -= WINDOW_WIDTH;
        if (pd.y < 0) pd.y += WINDOW_HEIGHT;
        else if (pd.y >= WINDOW_HEIGHT) pd.y -= WINDOW_HEIGHT;

        // Energy
        pd.energy -= ENERGY_DECAY;
        if (pd.energy <= 0.f) {
            pd.alive = false;
        }
    }

    // --- Prey ---
    for (auto& py : prey) {
        if (!py.alive) continue;

        // Nearest predator
        float bestDist = 999999.f;
        float dx = 0.f, dy = 0.f;
        for (auto& pr : predators) {
            if (!pr.alive) continue;
            float dist = std::hypot(py.x - pr.x, py.y - pr.y);
            if (dist < bestDist) {
                bestDist = dist;
                dx = pr.x - py.x;
                dy = pr.y - py.y;
            }
        }
        if (bestDist > 900000.f) {
            bestDist = 0.f;
            dx = 0.f;
            dy = 0.f;
        }
        float len = std::hypot(dx, dy) + 0.0001f;
        dx /= len;
        dy /= len;

        float distFrac = bestDist / (float)std::max(WINDOW_WIDTH, WINDOW_HEIGHT);
        float energyFrac = py.energy / 20.f;
        float inputArr[LSTM_INPUT_SIZE] = { distFrac, dx, dy, energyFrac };
        float outArr[LSTM_OUTPUT_SIZE];

        lstmForward(py.genome, inputArr, py.hiddenState, py.cellState, outArr);

        py.vx = outArr[0] * MAX_SPEED;
        py.vy = outArr[1] * MAX_SPEED;
        py.x += py.vx;
        py.y += py.vy;

        // Wrap
        if (py.x < 0) py.x += WINDOW_WIDTH;
        else if (py.x >= WINDOW_WIDTH) py.x -= WINDOW_WIDTH;
        if (py.y < 0) py.y += WINDOW_HEIGHT;
        else if (py.y >= WINDOW_HEIGHT) py.y -= WINDOW_HEIGHT;

        // Energy
        py.energy -= ENERGY_DECAY;
        if (py.energy <= 0.f) {
            py.alive = false;
        }
    }

    // --- Predator eats Prey ---
    for (auto& pd : predators) {
        if (!pd.alive) continue;
        for (auto& py : prey) {
            if (!py.alive) continue;
            float d = std::hypot(pd.x - py.x, pd.y - py.y);
            if (d < COLLISION_DISTANCE) {
                // Predator eats prey
                py.alive = false;
                pd.energy += PREDATOR_EAT_BONUS;
                pd.fitness += 5.f;
                totalPreyEaten++;
            }
        }
    }

    // --- Prey eats Food ---
    for (auto& py : prey) {
        if (!py.alive) continue;
        for (auto& f : food) {
            if (f.eaten) continue;
            float d = std::hypot(py.x - f.x, py.y - f.y);
            if (d < COLLISION_DISTANCE) {
                f.eaten = true;
                py.energy += PREY_FOOD_ENERGY;
                py.fitness += 2.f;
            }
        }
    }

    // Survival-based fitness increments
    for (auto& pd : predators) {
        if (pd.alive) {
            pd.fitness += 0.01f;
        }
    }
    for (auto& py : prey) {
        if (py.alive) {
            py.fitness += 0.02f;
        }
    }
}

void Simulation::logStats()
{
    // Compute average/max fitness for predators
    float sumPredFit = 0.f;
    float maxPredFit = 0.f;
    for (auto& pd : predators) {
        sumPredFit += pd.fitness;
        if (pd.fitness > maxPredFit) {
            maxPredFit = pd.fitness;
        }
    }
    float avgPredFit = sumPredFit / (float)predators.size();

    // Compute average/max fitness for prey
    float sumPreyFit = 0.f;
    float maxPreyFit = 0.f;
    for (auto& py : prey) {
        sumPreyFit += py.fitness;
        if (py.fitness > maxPreyFit) {
            maxPreyFit = py.fitness;
        }
    }
    float avgPreyFit = sumPreyFit / (float)prey.size();

    // Open stats file in append mode
    std::ofstream statsFile("stats.csv", std::ios::app);
    if (!statsFile.is_open()) {
        std::cerr << "Could not open stats.csv for writing!\n";
        return;
    }

    // Write header if not yet done
    if (!wroteHeader) {
        statsFile << "Gen,AvgPredFit,MaxPredFit,AvgPreyFit,MaxPreyFit,PreyEaten\n";
        wroteHeader = true;
    }

    // Write this generation's data
    statsFile << generation << ","
        << avgPredFit << ","
        << maxPredFit << ","
        << avgPreyFit << ","
        << maxPreyFit << ","
        << totalPreyEaten << "\n";

    statsFile.close();
}
