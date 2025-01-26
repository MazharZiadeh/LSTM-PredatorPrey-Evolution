#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>

// Global random engine & uniform distribution [0,1]
extern std::mt19937 rng;
extern std::uniform_real_distribution<float> dist01;

#endif // RANDOM_HPP
