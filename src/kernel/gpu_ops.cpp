#include <pybind11/pybind11.h>
#include "kernel.h"
#include "helpers.h"

template <typename T> pybind11::bytes PackDescriptor(const T& descriptor) {
	return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T* fn) {
	return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict TreeGPEvalRegistrations() {
    pybind11::dict dict;
    dict["gp_eval_forward"] =
        EncapsulateFunction(treeGP_eval);
    return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
    m.def("get_gp_eval_registrations", &TreeGPEvalRegistrations);
    m.def("create_gp_eval_descriptor",
        [](unsigned int popSize, unsigned int gpLen, unsigned int varLen, ElementType type) {
                return PackDescriptor(TreeGPEvalDescriptor{
                    popSize, gpLen, varLen, type });
        });

    pybind11::enum_<ElementType>(m, "ElementType")
        .value("BF16", ElementType::BF16)
        .value("F16", ElementType::F16)
        .value("F32", ElementType::F32)
        .value("F64", ElementType::F64);

}