#include "Random.hpp"

// Define once in a .cpp
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> dist01(0.f, 1.f);
