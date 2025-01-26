#ifndef LSTMN_NETWORK_HPP
#define LSTMN_NETWORK_HPP

#include <vector>

/**
 * Single-step LSTM forward:
 * - inputs[]: LSTM_INPUT_SIZE
 * - hidden & cell: updated in place
 * - outputs[2]: final movement vector in [-1,1]
 */
void lstmForward(const std::vector<float>& genome,
    const float* inputs,
    std::vector<float>& hidden,
    std::vector<float>& cell,
    float outputs[2]);

#endif // LSTMN_NETWORK_HPP
