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

pybind11::dict TreeGPSRFitnessRegistrations() {
	pybind11::dict dict;
	dict["gp_sr_fitness_forward"] =
		EncapsulateFunction(treeGP_SR_fitness);
	return dict;
}

pybind11::dict TreeGPGenerateRegistrations() {
	pybind11::dict dict;
	dict["gp_generate_forward"] =
		EncapsulateFunction(treeGP_generate);
	return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
	m.def("get_gp_eval_registrations", &TreeGPEvalRegistrations);
	m.def("get_gp_crossover_registrations", &TreeGPCorssoverRegistrations);
	m.def("get_gp_mutation_registrations", &TreeGPMutationRegistrations);
	m.def("get_gp_sr_fitness_registrations", &TreeGPSRFitnessRegistrations);
	m.def("get_gp_generate_registrations", &TreeGPGenerateRegistrations);
	m.def("create_gp_descriptor",
		[](int popSize, int gpLen, int varLen, int outLen, ElementType type)
		{
			return PackDescriptor(TreeGPDescriptor(popSize, gpLen, varLen, outLen, type));
		});
	m.def("create_gp_sr_descriptor",
		[](int popSize, int dataPoints, int gpLen, int varLen, int outLen, ElementType type, bool useMSE)
		{
			return PackDescriptor(TreeGPSRDescriptor(popSize, dataPoints, gpLen, varLen, outLen, type, useMSE));
		});
	m.def("create_gp_generate_descriptor",
		[](int popSize, int gpLen, int varLen, int outLen, int constSamplesLen, float outProb, float constProb, const RandomEngine engine, const ElementType type)
		{
			return PackDescriptor(TreeGPGenerateDescriptor(popSize, gpLen, varLen, outLen, constSamplesLen, outProb, constProb, engine, type));
		});

	pybind11::enum_<ElementType>(m, "ElementType")
		.value("BF16", ElementType::BF16)
		.value("F16", ElementType::F16)
		.value("F32", ElementType::F32)
		.value("F64", ElementType::F64);

	pybind11::enum_<RandomEngine>(m, "RandomEngine")
		.value("Default", RandomEngine::Default)
		.value("RANLUX24", RandomEngine::RANLUX24)
		.value("RANLUX48", RandomEngine::RANLUX48)
		.value("TAUS88", RandomEngine::TAUS88);
}