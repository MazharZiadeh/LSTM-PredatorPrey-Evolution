#include "LSTMNetwork.hpp"
#include "Constants.hpp"
#include <cmath>
#include <cassert>

/**
 * Multiply-add for each gate
 * offset => start in genome
 * input => size inSize
 * prevH => size hiddenSize
 * out => size hiddenSize
 */
static inline void matVecMulAddGate(const std::vector<float>& params,
    unsigned offset,
    const float* input,
    const float* prevH,
    unsigned inSize,
    unsigned hiddenSize,
    float* out)
{
    const unsigned weightCount = (inSize + hiddenSize) * hiddenSize;
    const unsigned biasOffset = offset + weightCount;

    // zero out
    for (unsigned h = 0; h < hiddenSize; h++) {
        out[h] = 0.f;
    }

    // input->hidden
    for (unsigned h = 0; h < hiddenSize; h++) {
        for (unsigned i = 0; i < inSize; i++) {
            float w = params[offset + i * hiddenSize + h];
            out[h] += input[i] * w;
        }
    }

    // hidden->hidden
    unsigned hiddenWeightsStart = offset + (inSize * hiddenSize);
    for (unsigned h = 0; h < hiddenSize; h++) {
        for (unsigned hh = 0; hh < hiddenSize; hh++) {
            float w = params[hiddenWeightsStart + hh * hiddenSize + h];
            out[h] += prevH[hh] * w;
        }
    }

    // add bias
    for (unsigned h = 0; h < hiddenSize; h++) {
        out[h] += params[biasOffset + h];
    }
}

void lstmForward(const std::vector<float>& genome,
    const float* inputs,
    std::vector<float>& hidden,
    std::vector<float>& cell,
    float outputs[2])
{
    assert(genome.size() == LSTM_GENOME_SIZE);

    // Parse gates: i, f, g, o
    unsigned offset = 0;

    float gateInput[LSTM_HIDDEN_SIZE];
    float gateForget[LSTM_HIDDEN_SIZE];
    float gateCell[LSTM_HIDDEN_SIZE];
    float gateOutput[LSTM_HIDDEN_SIZE];

    // Input gate (i)
    matVecMulAddGate(genome, offset, inputs, hidden.data(),
        LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, gateInput);
    offset += (LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE) * LSTM_HIDDEN_SIZE + LSTM_HIDDEN_SIZE;
    for (unsigned h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        gateInput[h] = 1.f / (1.f + std::exp(-gateInput[h])); // sigmoid
    }

    // Forget gate (f)
    matVecMulAddGate(genome, offset, inputs, hidden.data(),
        LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, gateForget);
    offset += (LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE) * LSTM_HIDDEN_SIZE + LSTM_HIDDEN_SIZE;
    for (unsigned h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        gateForget[h] = 1.f / (1.f + std::exp(-gateForget[h]));
    }

    // Cell gate (g)
    matVecMulAddGate(genome, offset, inputs, hidden.data(),
        LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, gateCell);
    offset += (LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE) * LSTM_HIDDEN_SIZE + LSTM_HIDDEN_SIZE;
    for (unsigned h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        gateCell[h] = std::tanh(gateCell[h]);
    }

    // Output gate (o)
    matVecMulAddGate(genome, offset, inputs, hidden.data(),
        LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, gateOutput);
    offset += (LSTM_INPUT_SIZE + LSTM_HIDDEN_SIZE) * LSTM_HIDDEN_SIZE + LSTM_HIDDEN_SIZE;
    for (unsigned h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        gateOutput[h] = 1.f / (1.f + std::exp(-gateOutput[h]));
    }

    // LSTM update
    for (unsigned h = 0; h < LSTM_HIDDEN_SIZE; h++) {
        cell[h] = gateForget[h] * cell[h] + gateInput[h] * gateCell[h];
        hidden[h] = gateOutput[h] * std::tanh(cell[h]);
    }

    // Final FC layer: hidden->2
    for (unsigned o = 0; o < LSTM_OUTPUT_SIZE; o++) {
        float sum = genome[offset + LSTM_FC_WEIGHTS + o]; // bias
        for (unsigned hh = 0; hh < LSTM_HIDDEN_SIZE; hh++) {
            float w = genome[offset + o * LSTM_HIDDEN_SIZE + hh];
            sum += hidden[hh] * w;
        }
        outputs[o] = std::tanh(sum);
    }
}
