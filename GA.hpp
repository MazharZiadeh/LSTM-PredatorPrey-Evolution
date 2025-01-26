#ifndef GA_HPP
#define GA_HPP

#include <vector>
#include "Creature.hpp"

// Genetic operators
std::vector<float> crossover(const std::vector<float>& p1, const std::vector<float>& p2);
void mutate(std::vector<float>& genome);

void evolvePopulation(std::vector<Creature>& pop, CreatureType t, unsigned newSize);

#endif
