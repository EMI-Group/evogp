#include "kernel.h"
#include <cmath>
#include <stdio.h>

// #include <thrust/execution_policy.h>
// #include <thrust/reduce.h>

// #undef CONST

__constant__ float const_node_value[MAX_STACK];
__constant__ int16_t const_node_type[MAX_STACK];
__constant__ int16_t const_tree_size;
__constant__ float const_variables[MAX_STACK];
__constant__ float const_labels[MAX_STACK];

int getCudaCoreCount() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int coresPerSM;
    switch (prop.major * 10 + prop.minor) {
    case 30:
        coresPerSM = 192;
        break; // Kepler
    case 32:
        coresPerSM = 192;
        break;
    case 35:
        coresPerSM = 192;
        break;
    case 37:
        coresPerSM = 192;
        break;
    case 50:
        coresPerSM = 128;
        break; // Maxwell
    case 52:
        coresPerSM = 128;
        break;
    case 53:
        coresPerSM = 128;
        break;
    case 60:
        coresPerSM = 64;
        break; // Pascal
    case 61:
        coresPerSM = 128;
        break;
    case 62:
        coresPerSM = 128;
        break;
    case 70:
        coresPerSM = 64;
        break; // Volta
    case 72:
        coresPerSM = 64;
        break;
    case 75:
        coresPerSM = 64;
        break; // Turing
    case 80:
        coresPerSM = 64;
        break; // Ampere
    case 86:
        coresPerSM = 128;
        break;
    case 87:
        coresPerSM = 128;
        break;
    case 89:
        coresPerSM = 128;
        break; // Ada Lovelace
    default:
        coresPerSM = 128;
        break; // defalut
    }
    return prop.multiProcessorCount * coresPerSM;
}

template <bool multiOutput = false>
__device__ inline void _process_node(
    int16_t node_type,
    float node_value,
    const int outLen,
    float *s_vars,
    float *s_vals, // stack
    float *s_outs,
    int &top) {
    // for multiOutput
    int16_t is_outNode = 0;
    float right_node = 0;
    if constexpr (multiOutput) {
        is_outNode = node_type & (int16_t)NodeType::OUT_NODE;
        node_type &= NodeType::TYPE_MASK;
    }
    // if the node is leaf
    if (node_type == NodeType::CONST) {
        s_vals[top++] = node_value;
        return;
    } else if (node_type == NodeType::VAR) {
        int var_num = (int)node_value;
        s_vals[top++] = s_vars[var_num];
        return;
    }

    // not a leaf, will be function
    unsigned int function, outIdx;
    function = (unsigned int)node_value;
    if constexpr (multiOutput) // value(float32) contains the function(int16_t) and outIndex(int16_t) info will using multiOutput mode
    {
        if (is_outNode) {
            OutNodeValue v = *(OutNodeValue *)&node_value;
            function = v.function;
            outIdx = v.outIndex;
        }
    }

    float top_val{};
    if (node_type == NodeType::UFUNC) {
        float var1 = s_vals[--top];

        if constexpr (multiOutput) {
            right_node = var1;
        }

        if (function == Function::SIN) {
            top_val = std::sin(var1);
        } else if (function == Function::COS) {
            top_val = std::cos(var1);
        } else if (function == Function::SINH) {
            top_val = std::sinh(var1);
        } else if (function == Function::TAN) {
            top_val = std::tan(var1);
        } else if (function == Function::COSH) {
            top_val = std::cosh(var1);
        } else if (function == Function::TANH) {
            top_val = std::tanh(var1);
        } else if (function == Function::LOG) {
            top_val = std::log(var1);
        } else if (function == Function::LOOSE_LOG) {
            if (var1 == 0.0f) {
                top_val = -MAX_VAL;
            } else {
                top_val = std::log(std::abs(var1));
            }
        } else if (function == Function::INV) {
            if (var1 == 0.0f) {
                top_val = MAX_VAL;
            } else
                top_val = 1.0f / var1;
        } else if (function == Function::LOOSE_INV) {
            if (std::abs(var1) <= DELTA) {
                var1 = copy_sign(DELTA, var1);
            }
            top_val = 1.0f / var1;
        } else if (function == Function::EXP) {
            top_val = std::exp(var1);
        } else if (function == Function::NEG) {
            top_val = -var1;
        } else if (function == Function::ABS) {
            top_val = std::abs(var1);
        } else if (function == Function::SQRT) {
            top_val = std::sqrt(var1);
        } else if (function == Function::LOOSE_SQRT) {
            if (var1 <= 0.0f) {
                var1 = std::abs(var1);
            }
            top_val = std::sqrt(var1);
        }
    } else if (node_type == NodeType::BFUNC) {
        float var1 = s_vals[--top];
        float var2 = s_vals[--top];

        if constexpr (multiOutput) {
            right_node = var2;
        }

        if (function == Function::ADD) {
            top_val = var1 + var2;
        } else if (function == Function::SUB) {
            top_val = var1 - var2;
        } else if (function == Function::MUL) {
            top_val = var1 * var2;
        } else if (function == Function::DIV) {
            if (var2 == 0.0f) {
                top_val = 1;
            } else
                top_val = var1 / var2;
        } else if (function == Function::LOOSE_DIV) {
            if (std::abs(var2) <= DELTA) {
                var2 = copy_sign(DELTA, var2);
            }
            top_val = var1 / var2;
        } else if (function == Function::POW) {
            top_val = std::pow(var1, var2);
        } else if (function == Function::LOOSE_POW) {
            if (var1 == 0.0f && var2 == 0.0f) {
                top_val = 0.0f;
            } else {
                top_val = std::pow(std::abs(var1), var2);
            }
        } else if (function == Function::MAX) {
            top_val = var1 >= var2 ? var1 : var2;
        } else if (function == Function::MIN) {
            top_val = var1 <= var2 ? var1 : var2;
        } else if (function == Function::LT) {
            top_val = var1 < var2 ? 1 : -1;
        } else if (function == Function::GT) {
            top_val = var1 > var2 ? 1 : -1;
        } else if (function == Function::LE) {
            top_val = var1 <= var2 ? 1 : -1;
        } else if (function == Function::GE) {
            top_val = var1 >= var2 ? 1 : -1;
        }
    } else //// if (node_type == NodeType::TFUNC)
    {
        float var1 = s_vals[--top];
        float var2 = s_vals[--top];
        float var3 = s_vals[--top];
        if constexpr (multiOutput) {
            right_node = var3;
        }
        //// if (function == Function::IF)
        top_val = var1 > (0.0f) ? var2 : var3;
    }

    // clip value
    // if (is_nan(top_val))
    // {
    // 	top_val = .0f;
    // }
    // else if (is_inf(top_val) || std::abs(top_val) > MAX_VAL)
    // {
    // 	top_val = copy_sign(MAX_VAL, top_val);
    // }

    // multiple output
    if constexpr (multiOutput) {
        // Y. Zhang and M. Zhang, “A multiple-output program tree structure ingenetic programming,” in Proceedings of. Citeseer, 2004
        if (is_outNode && outIdx < outLen)
            s_outs[outIdx] += top_val;
        top_val = right_node; // pass right_value to its father
    }
    s_vals[top++] = top_val;
}

template <bool multiOutput = false>
__device__ inline void _treeGPEvalByStack(
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *i_vars, // variables
    float *s_vals,       // stack, size = MAX_STACK_SIZE
    int16_t *s_infos,    // infos, size = 2 * MAX_STACK_SIZE
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    float *&s_outs,
    int &top) {
    /*
        s_vals: float*, stack memory stores the stack for operants. Also used to store tree values (avoid random access to global memory). length MAX_STACK_SIZE
        s_infos: int16_t*, stack memory stores some useful infos. length 2 * MAX_STACK_SIZE
            SingleOutput: [0, MAX_STACK_SIZE): tree types(int16_t), [MAX_STACK_SIZE, 2 * MAX_STACK_SIZE): variable values (float)
            MultiOutput: [0, MAX_
            STACK_SIZE): tree types(int16_t), [MAX_STACK_SIZE, 1.5 * MAX_STACK_SIZE): variable values (float), [1.5 * MAX_STACK_SIZE, 2 * MAX_STACK_SIZE): output values (float)
    */
    float *s_vars = (float *)(s_infos + MAX_STACK); // variable values on stack
    if constexpr (multiOutput)                      // outLen > 1, otherwise outLen = 1
    {
        s_outs = (float *)(s_infos + MAX_STACK + MAX_STACK / 2);
        for (int i = 0; i < outLen; i++) {
            s_outs[i] = 0; // output values on stack
        }
    }

    // load variable values from global memory to stack memory
    for (int i = 0; i < varLen; i++) {
        s_vars[i] = i_vars[i];
    }

    const unsigned int len = subtree_size[0];
    // load tree from global memory to stack memory
    // the order is: the inverse of prefix expression
    for (int i = 0; i < len; i++) {
        s_vals[len - 1 - i] = value[i];
        s_infos[len - 1 - i] = type[i];
    }

    // do stack operation according to the type of each node
    top = 0;
    for (int i = 0; i < len; i++) {
        // check node type
        int16_t node_type = s_infos[i];
        float node_value = s_vals[i];
        _process_node<multiOutput>(node_type, node_value, outLen, s_vars, s_vals, s_outs, top);
    }

    if (top != 1) {
        printf("top: %d\n", top);
    }
    assert(top == 1); // my personal guess
}

template <bool multiOutput = false>
__global__ void treeGPEvalKernel(
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    float *results) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= popSize)
        return;
    if constexpr (multiOutput) {
        assert(outLen > 1);
        assert(varLen <= MAX_STACK / 4);
        assert(outLen <= MAX_STACK / 4); // variable and outputs will load into infos
    } else {
        assert(varLen <= MAX_STACK / 2); // varible will load into infos
    }
    // init
    float *stack = (float *)alloca(MAX_STACK * sizeof(float));           // the stack to store the operants
    int16_t *infos = (int16_t *)alloca(2 * MAX_STACK * sizeof(int16_t)); // extra stack memory to load some info

    // current tree
    auto i_value = value + n * maxGPLen;
    auto i_type = type + n * maxGPLen;
    auto i_subtree_size = subtree_size + n * maxGPLen;

    // current variables
    auto i_vars = variables + n * varLen;

    // call
    float *s_outs{}; // output ptr. default is null. only used in multiOutput mode
    int top{};       // stack ptr.
    _treeGPEvalByStack<multiOutput>(i_value, i_type, i_subtree_size, i_vars, stack, infos, popSize, maxGPLen, varLen, outLen, s_outs, top);
    // final
    if constexpr (multiOutput) {
        // load s_outs in results
        auto o_res = results + n * outLen;
        for (int i = 0; i < outLen; i++) {
            o_res[i] = s_outs[i];
        }
    } else {
        results[n] = stack[--top]; // --top will always be 0? I guess
    }
}

void evaluate(
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    float *results) {
    int gridSize{}, blockSize{};
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPEvalKernel<false>);
    if (gridSize * blockSize < popSize)
        gridSize = (popSize - 1) / blockSize + 1;
    if (outLen > 1)
        treeGPEvalKernel<true><<<gridSize, blockSize>>>(popSize, maxGPLen, varLen, outLen, value, type, subtree_size, variables, results);
    else
        treeGPEvalKernel<false><<<gridSize, blockSize>>>(popSize, maxGPLen, varLen, 1, value, type, subtree_size, variables, results);
}

constexpr auto SR_BLOCK_SIZE = 1024;

template <bool multiOutput = false, bool useMSE = true>
__device__ inline float calculate_fit(
    int top,
    const float *stack,
    const float *s_outs,
    const float *i_labels,
    const unsigned int outLen = 1) {
    float fit = .0f;
    if constexpr (multiOutput) {
        for (int i = 0; i < outLen; i++) {
            float diff = i_labels[i] - s_outs[i];
            if constexpr (useMSE)
                fit += diff * diff;
            else
                fit += std::abs(diff); // abs
        }
    } else {
        float output_value = stack[--top];
        float diff = i_labels[0] - output_value;
        if constexpr (useMSE)
            fit = diff * diff;
        else
            fit = std::abs(diff);
    }
    return fit;
}

template <bool multiOutput = false, bool useMSE = true>
__global__ void treeGPRegressionFitnessKernel(
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    const float *labels,
    float *fitnesses, // len=(popSize, dataPoints)
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen = 1)
/**
 * gps: [popSize * maxLen]
 */
{
    const unsigned int block_cnt_per_tree = (dataPoints - 1) / blockDim.x + 1;
    const unsigned int nGP = blockIdx.x, nTB = blockIdx.y, threadId = threadIdx.x;
    const unsigned int dataPointId = nTB * blockDim.x + threadId;

    __shared__ float sharedFitness[SR_BLOCK_SIZE];

    if (nGP >= popSize || nTB >= block_cnt_per_tree)
        return;

    if constexpr (multiOutput) {
        assert(outLen > 1);
        assert(varLen * sizeof(float) / sizeof(int) <= MAX_STACK / 4);
        assert(outLen * sizeof(float) / sizeof(int) <= MAX_STACK / 4);
    } else {
        assert(varLen * sizeof(float) / sizeof(int) <= MAX_STACK / 2);
    }
    // init

    float fit = .0f;

    if (dataPointId < dataPoints) {
        float *stack = (float *)alloca(MAX_STACK * sizeof(float));
        int16_t *infos = (int16_t *)alloca(2 * MAX_STACK * sizeof(int16_t));

        // current tree
        auto i_value = value + nGP * maxGPLen;
        auto i_type = type + nGP * maxGPLen;
        auto i_subtree_size = subtree_size + nGP * maxGPLen;

        // evaluate over data points
        auto i_vars = variables + dataPointId * varLen;
        auto i_labels = labels + dataPointId * outLen;
        float *s_outs{};
        int top{};
        _treeGPEvalByStack<multiOutput>(i_value, i_type, i_subtree_size, i_vars, stack, infos, popSize, maxGPLen, varLen, outLen, s_outs, top);
        fit = calculate_fit<multiOutput, useMSE>(top, stack, s_outs, i_labels, outLen);
    }
    sharedFitness[threadId] = fit;

    __syncthreads();

    for (unsigned int size = SR_BLOCK_SIZE / 2; size > 0; size >>= 1) {
        if (threadId < size && threadId + size < blockDim.x) {
            sharedFitness[threadId] += sharedFitness[threadId + size];
        }
        // if (size >= 32)
        __syncthreads();
    }

    // only one atomicAdd in each block
    if (threadId == 0) {
        atomicAdd(&fitnesses[nGP], sharedFitness[0]);
    }
}

__global__ void averageFitnessValueKernel(float *fitnesses, const unsigned int popSize, const unsigned int dataPoints) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= popSize)
        return;
    fitnesses[n] /= dataPoints;
}

void normal_SR_fitness(
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int gpLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const bool useMSE,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    const float *labels,
    float *fitnesses) {
    const unsigned int threadBlocks = (dataPoints - 1) / SR_BLOCK_SIZE + 1; // number of blocks for one individual
    dim3 gridSize{popSize, threadBlocks};                                   // total blocks
    auto err = cudaMemsetAsync(fitnesses, 0, popSize * sizeof(float));      // clear fitnesses
    if (outLen > 1) {
        if (useMSE)
            treeGPRegressionFitnessKernel<true, true><<<gridSize, SR_BLOCK_SIZE>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
        else
            treeGPRegressionFitnessKernel<true, false><<<gridSize, SR_BLOCK_SIZE>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
    } else {
        if (useMSE)
            treeGPRegressionFitnessKernel<false, true><<<gridSize, SR_BLOCK_SIZE>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
        else
            treeGPRegressionFitnessKernel<false, false><<<gridSize, SR_BLOCK_SIZE>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
    }

    // average fitness value
    unsigned int averagethreadBlocks = (popSize - 1) / SR_BLOCK_SIZE + 1;
    averageFitnessValueKernel<<<averagethreadBlocks, SR_BLOCK_SIZE>>>(fitnesses, popSize, dataPoints);
}

void advanced_SR_fitness(
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int gpLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const bool useMSE,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    const float *labels,
    float *fitnesses) {
    auto err = cudaMemsetAsync(fitnesses, 0, popSize * sizeof(float)); // clear fitnesses
    unsigned int block_cnt_per_tree, block_size;
    block_size = dataPoints <= 256 ? dataPoints : SR_BLOCK_SIZE; // experience
    // block_size = dataPoints <= SR_BLOCK_SIZE ? dataPoints : SR_BLOCK_SIZE;

    block_cnt_per_tree = (dataPoints - 1) / block_size + 1;
    dim3 gridSize{popSize, block_cnt_per_tree}; // total blocks
    if (outLen > 1) {
        if (useMSE)
            treeGPRegressionFitnessKernel<true, true><<<gridSize, block_size>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
        else
            treeGPRegressionFitnessKernel<true, false><<<gridSize, block_size>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
    } else {
        if (useMSE)
            treeGPRegressionFitnessKernel<false, true><<<gridSize, block_size>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
        else
            treeGPRegressionFitnessKernel<false, false><<<gridSize, block_size>>>(value, type, subtree_size, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
    }

    // average fitness value
    unsigned int averagethreadBlocks = (popSize - 1) / SR_BLOCK_SIZE + 1;
    averageFitnessValueKernel<<<averagethreadBlocks, SR_BLOCK_SIZE>>>(fitnesses, popSize, dataPoints);
}

template <bool multiOutput = false>
__device__ inline void _constant_tree_treeGPEvalByStack(
    const float *i_vars, // variables
    float *s_vals,       // stack, size = MAX_STACK_SIZE
    int16_t *s_infos,    // infos, size = 2 * MAX_STACK_SIZE
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    float *&s_outs,
    int &top) {
    float *s_vars = (float *)(s_infos + MAX_STACK); // variable values on stack
    if constexpr (multiOutput)                      // outLen > 1, otherwise outLen = 1
    {
        s_outs = (float *)(s_infos + MAX_STACK + MAX_STACK / 2);
        for (int i = 0; i < outLen; i++) {
            s_outs[i] = 0; // output values on stack
        }
    }

    // load variable values from global memory to stack memory
    for (int i = 0; i < varLen; i++) {
        s_vars[i] = i_vars[i];
    }

    // do stack operation according to the type of each node
    top = 0;
    for (int i = const_tree_size - 1; i >= 0; --i) {
        // check node type
        int16_t node_type = const_node_type[i];
        float node_value = const_node_value[i];
        _process_node<multiOutput>(node_type, node_value, outLen, s_vars, s_vals, s_outs, top);
    }

    if (top != 1) {
        printf("top: %d\n", top);
    }
    assert(top == 1); // my personal guess
}

template <bool multiOutput = false, bool useMSE = true>
__global__ void constant_tree_treeGPRegressionFitnessKernel(
    const int treeId,
    const float *variables,
    const float *labels,
    float *fitnesses,
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen = 1) {
    const unsigned int dataPointId = blockIdx.x * SR_BLOCK_SIZE + threadIdx.x, threadId = threadIdx.x;
    __shared__ float sharedFitness[SR_BLOCK_SIZE];
    sharedFitness[threadId] = .0f;

    if (dataPointId >= dataPoints)
        return;

    if constexpr (multiOutput) {
        assert(outLen > 1);
        assert(varLen * sizeof(float) / sizeof(int) <= MAX_STACK / 4);
        assert(outLen * sizeof(float) / sizeof(int) <= MAX_STACK / 4);
    } else {
        assert(varLen * sizeof(float) / sizeof(int) <= MAX_STACK / 2);
    }

    // init
    float fit = .0f;
    float *stack = (float *)alloca(MAX_STACK * sizeof(float));
    int16_t *infos = (int16_t *)alloca(2 * MAX_STACK * sizeof(int16_t));

    // current data point
    auto i_vars = variables + dataPointId * varLen;
    auto i_labels = labels + dataPointId * outLen;

    float *s_outs{};
    int top{};
    _constant_tree_treeGPEvalByStack<multiOutput>(i_vars, stack, infos, popSize, maxGPLen, varLen, outLen, s_outs, top);
    fit = calculate_fit<multiOutput, useMSE>(top, stack, s_outs, i_labels, outLen);

    sharedFitness[threadId] = fit;

    __syncthreads();

    for (unsigned int size = SR_BLOCK_SIZE / 2; size > 0; size >>= 1) {
        if (threadId < size) {
            sharedFitness[threadId] += sharedFitness[threadId + size];
        }
        // if (size >= 32)
        __syncthreads();
    }

    // only one atomicAdd in each block
    if (threadId == 0) {
        atomicAdd(&fitnesses[treeId], sharedFitness[0]);
    }
}

void constant_tree_SR_fitness(
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int gpLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const bool useMSE,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    const float *labels,
    float *fitnesses) {
    cudaMemsetAsync(fitnesses, 0, popSize * sizeof(float));                 // clear fitnesses
    const unsigned int threadBlocks = (dataPoints - 1) / SR_BLOCK_SIZE + 1; // number of blocks for one individual
    for (int i = 0; i < popSize; i++) {
        // copy tree to constant memory
        auto current_node_value = value + i * gpLen;
        auto current_node_type = type + i * gpLen;
        auto current_subtree_size = subtree_size + i * gpLen;

        cudaMemcpyToSymbolAsync(const_node_value, current_node_value, gpLen * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbolAsync(const_node_type, current_node_type, gpLen * sizeof(int16_t), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbolAsync(const_tree_size, current_subtree_size, sizeof(int16_t), 0, cudaMemcpyDeviceToDevice);

        if (outLen > 1) {
            if (useMSE)
                constant_tree_treeGPRegressionFitnessKernel<true, true><<<threadBlocks, SR_BLOCK_SIZE>>>(i, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
            else
                constant_tree_treeGPRegressionFitnessKernel<true, false><<<threadBlocks, SR_BLOCK_SIZE>>>(i, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
        } else {
            if (useMSE)
                constant_tree_treeGPRegressionFitnessKernel<false, true><<<threadBlocks, SR_BLOCK_SIZE>>>(i, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
            else
                constant_tree_treeGPRegressionFitnessKernel<false, false><<<threadBlocks, SR_BLOCK_SIZE>>>(i, variables, labels, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
        }
    }
    // average fitness value
    int gridSize = 0, blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, averageFitnessValueKernel);
    if (gridSize * blockSize < popSize)
        gridSize = (popSize - 1) / blockSize + 1;
    averageFitnessValueKernel<<<gridSize, blockSize>>>(fitnesses, popSize, dataPoints);
}

template <bool multiOutput = false>
__device__ inline void _constant_vars_treeGPEvalByStack(
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    float *s_vals,    // stack, size = MAX_STACK_SIZE
    int16_t *s_infos, // infos, size = 2 * MAX_STACK_SIZE
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    float *&s_outs,
    int &top) {
    if constexpr (multiOutput) // outLen > 1, otherwise outLen = 1
    {
        s_outs = (float *)(s_infos + MAX_STACK + MAX_STACK / 2);
        for (int i = 0; i < outLen; i++) {
            s_outs[i] = 0; // output values on stack
        }
    }

    const unsigned int len = subtree_size[0];
    // load tree from global memory to stack memory
    // the order is: the inverse of prefix expression
    for (int i = 0; i < len; i++) {
        s_vals[len - 1 - i] = value[i];
        s_infos[len - 1 - i] = type[i];
    }

    // do stack operation according to the type of each node
    top = 0;
    for (int i = 0; i < len; i++) {
        // check node type
        int16_t node_type = s_infos[i];
        float node_value = s_vals[i];
        _process_node<multiOutput>(node_type, node_value, outLen, const_variables, s_vals, s_outs, top);
    }

    if (top != 1) {
        printf("top: %d\n", top);
    }
    assert(top == 1); // my personal guess
}

template <bool multiOutput = false, bool useMSE = true>
__global__ void constant_vars_treeGPRegressionFitnessKernel(
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    float *fitnesses,
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen = 1) {
    const unsigned int treeId = blockIdx.x * blockDim.x + threadIdx.x;

    if (treeId >= popSize)
        return;

    if constexpr (multiOutput) {
        assert(outLen > 1);
        assert(varLen * sizeof(float) / sizeof(int) <= MAX_STACK / 4);
        assert(outLen * sizeof(float) / sizeof(int) <= MAX_STACK / 4);
    } else {
        assert(varLen * sizeof(float) / sizeof(int) <= MAX_STACK / 2);
    }

    // init
    float fit = .0f;
    float *stack = (float *)alloca(MAX_STACK * sizeof(float));
    int16_t *infos = (int16_t *)alloca(2 * MAX_STACK * sizeof(int16_t));

    // current tree
    auto i_value = value + treeId * maxGPLen;
    auto i_type = type + treeId * maxGPLen;
    auto i_subtree_size = subtree_size + treeId * maxGPLen;

    float *s_outs{};
    int top{};
    _constant_vars_treeGPEvalByStack<multiOutput>(i_value, i_type, i_subtree_size, stack, infos, popSize, maxGPLen, varLen, outLen, s_outs, top);
    fit = calculate_fit<multiOutput, useMSE>(top, stack, s_outs, const_labels, outLen);

    // there are no conflicts at all, so no need to lock
    fitnesses[treeId] += fit;
}

void constant_vars_SR_fitness(
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int gpLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const bool useMSE,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    const float *labels,
    float *fitnesses) {
    int gridSize = 0, blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, constant_vars_treeGPRegressionFitnessKernel<false, true>);
    if (gridSize * blockSize < popSize)
        gridSize = (popSize - 1) / blockSize + 1;

    cudaMemsetAsync(fitnesses, 0, popSize * sizeof(float)); // clear fitnesses
    for (int i = 0; i < dataPoints; i++) {
        // copy vars and labels to constant memory
        auto current_variables = variables + i * varLen;
        auto current_labels = labels + i * outLen;

        cudaMemcpyToSymbolAsync(const_variables, current_variables, varLen * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbolAsync(const_labels, current_labels, outLen * sizeof(float), 0, cudaMemcpyDeviceToDevice);

        if (outLen > 1) {
            if (useMSE)
                constant_vars_treeGPRegressionFitnessKernel<true, true><<<gridSize, blockSize>>>(value, type, subtree_size, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
            else
                constant_vars_treeGPRegressionFitnessKernel<true, false><<<gridSize, blockSize>>>(value, type, subtree_size, fitnesses, popSize, dataPoints, gpLen, varLen, outLen);
        } else {
            if (useMSE)
                constant_vars_treeGPRegressionFitnessKernel<false, true><<<gridSize, blockSize>>>(value, type, subtree_size, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
            else
                constant_vars_treeGPRegressionFitnessKernel<false, false><<<gridSize, blockSize>>>(value, type, subtree_size, fitnesses, popSize, dataPoints, gpLen, varLen, 1);
        }
    }
    // average fitness value
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, averageFitnessValueKernel);
    if (gridSize * blockSize < popSize)
        gridSize = (popSize - 1) / blockSize + 1;
    averageFitnessValueKernel<<<gridSize, blockSize>>>(fitnesses, popSize, dataPoints);
}

void SR_fitness(
    const unsigned int popSize,
    const unsigned int dataPoints,
    const unsigned int gpLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const bool useMSE,
    const float *value,
    const int16_t *type,
    const int16_t *subtree_size,
    const float *variables,
    const float *labels,
    float *fitnesses,
    const unsigned int kernel_type) {
    if (kernel_type == 0) {
        normal_SR_fitness(popSize, dataPoints, gpLen, varLen, outLen, useMSE, value, type, subtree_size, variables, labels, fitnesses);
    } else if (kernel_type == 1) {
        constant_tree_SR_fitness(popSize, dataPoints, gpLen, varLen, outLen, useMSE, value, type, subtree_size, variables, labels, fitnesses);
    } else if (kernel_type == 2) {
        constant_vars_SR_fitness(popSize, dataPoints, gpLen, varLen, outLen, useMSE, value, type, subtree_size, variables, labels, fitnesses);
    } else if (kernel_type == 3) {
        advanced_SR_fitness(popSize, dataPoints, gpLen, varLen, outLen, useMSE, value, type, subtree_size, variables, labels, fitnesses);
    } else if (kernel_type == 4) { // auto execute
        if (dataPoints <= getCudaCoreCount()) {
            advanced_SR_fitness(popSize, dataPoints, gpLen, varLen, outLen, useMSE, value, type, subtree_size, variables, labels, fitnesses);
        } else {
            constant_tree_SR_fitness(popSize, dataPoints, gpLen, varLen, outLen, useMSE, value, type, subtree_size, variables, labels, fitnesses);
        }
    }
}
