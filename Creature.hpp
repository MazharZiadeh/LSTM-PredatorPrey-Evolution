#ifndef CREATURE_HPP
#define CREATURE_HPP

#include <vector>

enum class CreatureType { PREDATOR, PREY };

struct Creature {
    CreatureType type;
    float x, y;
    float vx, vy;
    std::vector<float> genome; // LSTM parameters
    float fitness;
    bool  alive;
    float energy;

    // LSTM hidden & cell states
    std::vector<float> hiddenState;
    std::vector<float> cellState;

    // Constructor
    Creature(CreatureType t, float px, float py, const std::vector<float>& g);
};

#endif
