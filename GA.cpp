#include "GA.hpp"
#include "Constants.hpp"
#include "Random.hpp"
#include <algorithm>
#include <cmath>

std::vector<float> crossover(const std::vector<float>& p1, const std::vector<float>& p2)
{
    // Uniform crossover
    std::vector<float> child(p1.size());
    for (size_t i = 0; i < p1.size(); i++) {
        child[i] = (dist01(rng) < 0.5f) ? p1[i] : p2[i];
    }
    return child;
}

void mutate(std::vector<float>& genome)
{
    // Each gene has chance to be perturbed
    for (float& g : genome) {
        if (dist01(rng) < MUTATION_RATE) {
            float delta = (dist01(rng) * 2.f - 1.f)
                * MUTATION_POWER * (0.1f + std::fabs(g));
            g += delta;
        }
    }
}

void evolvePopulation(std::vector<Creature>& pop, CreatureType t, unsigned newSize)
{
    // Sort descending by fitness
    std::sort(pop.begin(), pop.end(), [](auto& a, auto& b) {
        return a.fitness > b.fitness;
        });

    std::vector<Creature> newPop;
    newPop.reserve(newSize);

    // Elitism
    unsigned eliteCount = std::min(ELITE_COUNT, (unsigned)pop.size());
    for (unsigned i = 0; i < eliteCount && i < newSize; i++) {
        pop[i].fitness = 0.f;
        pop[i].alive = true;
        pop[i].energy = INITIAL_ENERGY;
        // reset positions
        pop[i].x = dist01(rng) * WINDOW_WIDTH;
        pop[i].y = dist01(rng) * WINDOW_HEIGHT;
        // reset LSTM states
        std::fill(pop[i].hiddenState.begin(), pop[i].hiddenState.end(), 0.f);
        std::fill(pop[i].cellState.begin(), pop[i].cellState.end(), 0.f);

        newPop.push_back(pop[i]);
    }

    // Fill the remainder with crossover+mutation
    while (newPop.size() < newSize) {
        unsigned p1Idx = (unsigned)(dist01(rng) * (pop.size() / 2));
        unsigned p2Idx = (unsigned)(dist01(rng) * (pop.size() / 2));

        auto childGenome = crossover(pop[p1Idx].genome, pop[p2Idx].genome);
        mutate(childGenome);

        Creature child(t,
            dist01(rng) * WINDOW_WIDTH,
            dist01(rng) * WINDOW_HEIGHT,
            childGenome);
        newPop.push_back(child);
    }

    pop = newPop;
}
