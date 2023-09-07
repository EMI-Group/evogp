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

pybind11::dict TreeGPCorssoverRegistrations() {
	pybind11::dict dict;
	dict["gp_crossover_forward"] =
		EncapsulateFunction(treeGP_crossover);
	return dict;
}

pybind11::dict TreeGPMutationRegistrations() {
	pybind11::dict dict;
	dict["gp_mutation_forward"] =
		EncapsulateFunction(treeGP_mutation);
	return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
	m.def("get_gp_eval_registrations", &TreeGPEvalRegistrations);
	m.def("get_gp_crossover_registrations", &TreeGPCorssoverRegistrations);
	m.def("get_gp_mutation_registrations", &TreeGPMutationRegistrations);
	m.def("create_gp_descriptor",
		[](unsigned int popSize, unsigned int gpLen, unsigned int varLen, ElementType type)
		{
			return PackDescriptor(TreeGPDescriptor{popSize, gpLen, varLen, type});
		});

	pybind11::enum_<ElementType>(m, "ElementType")
		.value("BF16", ElementType::BF16)
		.value("F16", ElementType::F16)
		.value("F32", ElementType::F32)
		.value("F64", ElementType::F64);
}