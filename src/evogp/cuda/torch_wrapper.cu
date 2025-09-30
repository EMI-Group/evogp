#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>
#include "kernel.h"


void check_tensor(
    const torch::Tensor& tensor,
    c10::IntArrayRef expected_shape,
    const std::string& tensor_name
) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), tensor_name, " must be a contiguous CUDA tensor");
    TORCH_CHECK( 
        tensor.sizes() == expected_shape,
        tensor_name, " must have shape ", expected_shape, ", but got shape ", tensor.sizes()
    );
}

void check_cuda_error(const std::string& error_func_name){
    // Check for CUDA kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err), " in ", error_func_name);
    }

    // Synchronize the device and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel execution failed: ", cudaGetErrorString(err), " in ", error_func_name);
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tree_generate(
    int64_t pop_size,
    int64_t gp_len,
    int64_t var_len,
    int64_t out_len,
    int64_t const_samples_len,
    double out_prob,
    double const_prob,
    torch::Tensor keys,
    torch::Tensor depth2leaf_probs,
    torch::Tensor roulette_funcs,
    torch::Tensor const_samples
) {
    // check parameters
    TORCH_CHECK(pop_size > 0, "pop_size must larger than 0, but got ", pop_size);
    TORCH_CHECK(0 < gp_len && gp_len <= MAX_STACK, "gp_len must be in range (0, ", MAX_STACK, "], but got ", gp_len);
    TORCH_CHECK(0 < var_len, "var_len must larger than 0, but got ", var_len);
    TORCH_CHECK(0 < out_len, "out_len must larger than 0, but got ", out_len);
    TORCH_CHECK(0 < const_samples_len, "const_samples_len must larger than 0, but got ", const_samples_len);
    TORCH_CHECK(0 <= out_prob && out_prob <= 1, "out_prob must be in range [0, 1], but got ", out_prob);
    TORCH_CHECK(0 <= const_prob && const_prob <= 1, "const_prob must be in range [0, 1], but got ", const_prob);

    // check tensor
    check_tensor(keys, {2}, "keys");
    check_tensor(depth2leaf_probs, {MAX_FULL_DEPTH}, "depth2leaf_probs");
    check_tensor(roulette_funcs, {Function::END}, "roulette_funcs");
    check_tensor(const_samples, {const_samples_len}, "const_samples");

    // create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(keys.device()).requires_grad(false);
    auto value_tensor = torch::empty({pop_size, gp_len}, options);
    auto node_type_tensor = torch::empty({pop_size, gp_len}, options.dtype(torch::kInt16));
    auto subtree_size_tensor = torch::empty({pop_size, gp_len}, options.dtype(torch::kInt16));

    generate(
        pop_size,
        gp_len,
        var_len,
        out_len,
        const_samples_len,
        out_prob,
        const_prob,
        keys.data_ptr<unsigned int>(), 
        depth2leaf_probs.data_ptr<float>(),
        roulette_funcs.data_ptr<float>(),
        const_samples.data_ptr<float>(), 
        value_tensor.data_ptr<float>(), 
        node_type_tensor.data_ptr<int16_t>(), 
        subtree_size_tensor.data_ptr<int16_t>()
    );
    // check_cuda_error("generate");

    // return multiple tensors
    return std::make_tuple(value_tensor, node_type_tensor, subtree_size_tensor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tree_mutate(
    int64_t pop_size,
    int64_t gp_len,
    torch::Tensor value_ori,
    torch::Tensor type_ori,
    torch::Tensor subtree_size_ori,
    torch::Tensor mutateIndices,
    torch::Tensor value_new,
    torch::Tensor type_new,
    torch::Tensor subtree_size_new)
{

    // check parameters
    TORCH_CHECK(pop_size > 0, "pop_size must larger than 0, but got ", pop_size);
    TORCH_CHECK(0 < gp_len && gp_len <= MAX_STACK, "gp_len must be in range (0, ", MAX_STACK, "], but got ", gp_len);

    // check tensor
    check_tensor(value_ori, {pop_size, gp_len}, "value_ori");
    check_tensor(type_ori, {pop_size, gp_len}, "type_ori");
    check_tensor(subtree_size_ori, {pop_size, gp_len}, "subtree_size_ori");
    check_tensor(mutateIndices, {pop_size}, "mutateIndices");
    check_tensor(value_new, {pop_size, gp_len}, "value_new");
    check_tensor(type_new, {pop_size, gp_len}, "type_new");
    check_tensor(subtree_size_new, {pop_size, gp_len}, "subtree_size_new");

    // create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(value_new.device()).requires_grad(false);
    auto value_res = torch::empty({pop_size, gp_len}, options);
    auto node_type_res = torch::empty({pop_size, gp_len}, options.dtype(torch::kInt16));
    auto subtree_size_res = torch::empty({pop_size, gp_len}, options.dtype(torch::kInt16));

    mutate(
        pop_size,
        gp_len,
        value_ori.data_ptr<float>(),
        type_ori.data_ptr<int16_t>(),
        subtree_size_ori.data_ptr<int16_t>(),
        mutateIndices.data_ptr<int>(),
        value_new.data_ptr<float>(),
        type_new.data_ptr<int16_t>(),
        subtree_size_new.data_ptr<int16_t>(),
        value_res.data_ptr<float>(),
        node_type_res.data_ptr<int16_t>(),
        subtree_size_res.data_ptr<int16_t>()
    );
    // check_cuda_error("mutate");
    // return multiple tensors
    return std::make_tuple(value_res, node_type_res, subtree_size_res);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tree_crossover(
    int64_t pop_size_ori,
    int64_t pop_size_new,
    int64_t gp_len,
    torch::Tensor value_ori,
    torch::Tensor type_ori,
    torch::Tensor subtree_size_ori,
    torch::Tensor left_idx,
    torch::Tensor right_idx,
    torch::Tensor left_node_idx,
    torch::Tensor right_node_idx
){

    // check parameters
    TORCH_CHECK(pop_size_ori > 0, "pop_size_ori must larger than 0, but got ", pop_size_ori);
    TORCH_CHECK(pop_size_new > 0, "pop_size_new must larger than 0, but got ", pop_size_new);
    TORCH_CHECK(0 < gp_len && gp_len <= MAX_STACK, "gp_len must be in range (0, ", MAX_STACK, "], but got ", gp_len);

    // check tensor
    check_tensor(value_ori, {pop_size_ori, gp_len}, "value_ori");
    check_tensor(type_ori, {pop_size_ori, gp_len}, "type_ori");
    check_tensor(subtree_size_ori, {pop_size_ori, gp_len}, "subtree_size_ori");
    check_tensor(left_idx, {pop_size_new}, "left_idx");
    check_tensor(right_idx, {pop_size_new}, "right_idx");
    check_tensor(left_node_idx, {pop_size_new}, "left_node_idx");
    check_tensor(right_node_idx, {pop_size_new}, "right_node_idx");

    // create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(value_ori.device()).requires_grad(false);
    auto value_res = torch::empty({pop_size_new, gp_len}, options);
    auto node_type_res = torch::empty({pop_size_new, gp_len}, options.dtype(torch::kInt16));
    auto subtree_size_res = torch::empty({pop_size_new, gp_len}, options.dtype(torch::kInt16));

    crossover(
        pop_size_ori,
        pop_size_new,
        gp_len,
        value_ori.data_ptr<float>(),
        type_ori.data_ptr<int16_t>(),
        subtree_size_ori.data_ptr<int16_t>(),
        left_idx.data_ptr<int>(),
        right_idx.data_ptr<int>(),
        left_node_idx.data_ptr<int>(),
        right_node_idx.data_ptr<int>(),
        value_res.data_ptr<float>(),
        node_type_res.data_ptr<int16_t>(),
        subtree_size_res.data_ptr<int16_t>()
    );
    // check_cuda_error("crossover");
    // return multiple tensors
    return std::make_tuple(value_res, node_type_res, subtree_size_res);
}

torch::Tensor tree_evaluate(
    int64_t pop_size,
    int64_t gp_len,
    int64_t var_len,
    int64_t out_len,
    torch::Tensor value,
    torch::Tensor type,
    torch::Tensor subtree_size,
    torch::Tensor variables
){

    // check parameters
    TORCH_CHECK(pop_size > 0, "pop_size must larger than 0, but got ", pop_size);
    TORCH_CHECK(0 < gp_len && gp_len <= MAX_STACK, "gp_len must be in range (0, ", MAX_STACK, "], but got ", gp_len);
    TORCH_CHECK(0 < var_len, "var_len must larger than 0, but got ", var_len);
    TORCH_CHECK(0 < out_len, "out_len must larger than 0, but got ", out_len);

    // check tensor
    check_tensor(value, {pop_size, gp_len}, "value");
    check_tensor(type, {pop_size, gp_len}, "type");
    check_tensor(subtree_size, {pop_size, gp_len}, "subtree_size");
    check_tensor(variables, {pop_size, var_len}, "variables");

    // create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(value.device()).requires_grad(false);
    auto results = torch::empty({pop_size, out_len}, options);

    evaluate(
        pop_size,
        gp_len,
        var_len,
        out_len,
        value.data_ptr<float>(),
        type.data_ptr<int16_t>(),
        subtree_size.data_ptr<int16_t>(),
        variables.data_ptr<float>(),
        results.data_ptr<float>()
    );
    // check_cuda_error("evaluate");
    return results;
}

torch::Tensor tree_SR_fitness(
    int64_t pop_size,
    int64_t data_points,
    int64_t gp_len,
    int64_t var_len,
    int64_t out_len,
    bool useMSE,
    torch::Tensor value,
    torch::Tensor type,
    torch::Tensor subtree_size,
    torch::Tensor variables,
    torch::Tensor labels,
    int64_t kernel_type = 4
){
    // check parameters
    TORCH_CHECK(pop_size > 0, "pop_size must larger than 0, but got ", pop_size);
    TORCH_CHECK(0 < gp_len && gp_len <= MAX_STACK, "gp_len must be in range (0, ", MAX_STACK, "], but got ", gp_len);
    TORCH_CHECK(0 < var_len, "var_len must larger than 0, but got ", var_len);
    TORCH_CHECK(0 < out_len, "out_len must larger than 0, but got ", out_len);
    TORCH_CHECK(0 < data_points, "data_points must larger than 0, but got ", data_points);

    // check tensor
    check_tensor(value, {pop_size, gp_len}, "value");
    check_tensor(type, {pop_size, gp_len}, "type");
    check_tensor(subtree_size, {pop_size, gp_len}, "subtree_size");
    check_tensor(variables, {data_points, var_len}, "variables");
    check_tensor(labels, {data_points, out_len}, "labels");

    // create output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(value.device()).requires_grad(false);
    auto fitness = torch::empty({pop_size}, options);

    SR_fitness(
        pop_size,
        data_points,
        gp_len,
        var_len,
        out_len,
        useMSE,
        value.data_ptr<float>(),
        type.data_ptr<int16_t>(),
        subtree_size.data_ptr<int16_t>(),
        variables.data_ptr<float>(),
        labels.data_ptr<float>(),
        fitness.data_ptr<float>(),
        kernel_type
    );
    // check_cuda_error("SR_fitness");
    return fitness;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("tree_generate", &tree_generate, "Tree Generate Function");
}

TORCH_LIBRARY(evogp_cuda, m) {
   // Note that "float" in the schema corresponds to the C++ double type
   // and the Python float type.
    m.def("tree_generate(int i1, int i2, int i3, int i4, int i5, float f1, float f2, Tensor t1, Tensor t2, Tensor t3, Tensor t4) -> (Tensor t5, Tensor t6, Tensor t7)");
    m.def("tree_mutate(int i1, int i2, Tensor t1, Tensor t2, Tensor t3, Tensor t4, Tensor t5, Tensor t6, Tensor t7) -> (Tensor t8, Tensor t9, Tensor t10)");
    m.def("tree_crossover(int i1, int i2, int i3, Tensor t1, Tensor t2, Tensor t3, Tensor t4, Tensor t5, Tensor t6, Tensor t7) -> (Tensor t8, Tensor t9, Tensor t10)");
    m.def("tree_evaluate(int i1, int i2, int i3, int i4, Tensor t1, Tensor t2, Tensor t3, Tensor t4) -> Tensor t5");
    m.def("tree_SR_fitness(int i1, int i2, int i3, int i4, int i5, bool b1, Tensor t1, Tensor t2, Tensor t3, Tensor t4, Tensor t5, int i6) -> Tensor t6");
}

TORCH_LIBRARY_IMPL(evogp_cuda, CUDA, m) {
    m.impl("tree_generate", &tree_generate);
    m.impl("tree_mutate", &tree_mutate);
    m.impl("tree_crossover", &tree_crossover);
    m.impl("tree_evaluate", &tree_evaluate);
    m.impl("tree_SR_fitness", &tree_SR_fitness);
}
