#include "Creature.hpp"
#include "Constants.hpp"

Creature::Creature(CreatureType t, float px, float py, const std::vector<float>& g)
    : type(t),
    x(px), y(py),
    vx(0.f), vy(0.f),
    genome(g),
    fitness(0.f),
    alive(true),
    energy(INITIAL_ENERGY),
    hiddenState(LSTM_HIDDEN_SIZE, 0.f),
    cellState(LSTM_HIDDEN_SIZE, 0.f)
{
    // start LSTM states at 0
}
