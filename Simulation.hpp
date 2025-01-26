#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <SFML/Graphics.hpp>
#include <vector>
#include <fstream> // for file output

#include "Creature.hpp"

struct Food {
    float x, y;
    bool  eaten;
};

class Simulation {
public:
    Simulation();
    void run();

private:
    sf::RenderWindow window;
    sf::Font font;
    sf::Text genText;

    unsigned generation;
    unsigned steps;

    std::vector<Creature> predators;
    std::vector<Creature> prey;
    std::vector<Food>     food;

    // Data logging
    int totalPreyEaten;  // how many prey eaten this generation
    bool wroteHeader;     // have we written the CSV header?

    void init();
    void handleEvents();
    void update();
    void render();
    void nextGeneration();

    void simulationStep();
    void createFood(unsigned count);

    // Logs stats to "stats.csv"
    void logStats();
};

#endif
