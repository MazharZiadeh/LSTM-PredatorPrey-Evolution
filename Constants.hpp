#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// Window & timing
static const unsigned WINDOW_WIDTH = 1200;
static const unsigned WINDOW_HEIGHT = 800;
static const unsigned FRAMERATE = 60;

// Population sizes
static const unsigned NUM_PREDATORS = 30;
static const unsigned NUM_PREY = 50;

// GA / Evolution
static const unsigned STEPS_PER_GENERATION = 1500;
static const float    MUTATION_RATE = 0.2f;
static const float    MUTATION_POWER = 0.2f;
static const unsigned ELITE_COUNT = 8;

// Simulation mechanics
static const float COLLISION_DISTANCE = 12.f;
static const float MAX_SPEED = 5.f;
static const float ENERGY_DECAY = 0.002f;
static const float PREDATOR_EAT_BONUS = 5.0f;
static const float PREY_FOOD_ENERGY = 2.0f;
static const float INITIAL_ENERGY = 10.0f;

//---------------------------------------------------
// LSTM Architecture: 4 inputs -> 8 hidden -> 2 outputs
//
// Single LSTM layer => 4 gates + final FC => 434 parameters total.
// We'll call it LSTM_GENOME_SIZE = 434
//---------------------------------------------------
static const unsigned LSTM_INPUT_SIZE = 4;
static const unsigned LSTM_HIDDEN_SIZE = 8;
static const unsigned LSTM_OUTPUT_SIZE = 2;

// Gate param: (IN + H)*H + H (bias) => For 4 gates
static const unsigned LSTM_GATE_PARAMS = (LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE) * LSTM_HIDDEN_SIZE
+ LSTM_HIDDEN_SIZE;
static const unsigned LSTM_LAYER_PARAMS = 4 * LSTM_GATE_PARAMS;

// Final FC from hidden->2 plus bias (16+2=18)
static const unsigned LSTM_FC_WEIGHTS = LSTM_HIDDEN_SIZE * LSTM_OUTPUT_SIZE;
static const unsigned LSTM_FC_BIASES = LSTM_OUTPUT_SIZE;

static const unsigned LSTM_GENOME_SIZE = LSTM_LAYER_PARAMS + LSTM_FC_WEIGHTS + LSTM_FC_BIASES;
// = 416 + 18 = 434

#endif // CONSTANTS_HPP
