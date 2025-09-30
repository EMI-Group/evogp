import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/script.h>

at::Tensor mymuladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
}

TORCH_LIBRARY(extension_cpp, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    m.impl("mymuladd", &mymuladd_cpu);
}
"""

load_inline(
    name='extension_cpp',
    cpp_sources=[cpp_source],
    is_python_module=False,
    verbose=True
)

a = torch.randn(3, dtype=torch.float32).rename(None)
b = torch.randn(3, dtype=torch.float32).rename(None)

result = torch.ops.extension_cpp.mymuladd(a, b, 1.0)
print("Input a:", a)
print("Input b:", b)
print("Result:", result)