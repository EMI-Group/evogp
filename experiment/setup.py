import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.environ["MAX_JOBS"] = "16"

setup(
    name="evogp",
    ext_modules=[
        CUDAExtension(
            name="evogp_cuda",
            sources=[
                "../src/evogp/cuda/torch_wrapper.cu",
                "../src/evogp/cuda/generate.cu",
                "../src/evogp/cuda/mutation.cu",
                "../src/evogp/cuda/forward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "--ptxas-options=-v",
                    "-Xptxas=-O3",
                    "-lineinfo",
                    "-lcudart",
                    "-use_fast_math",
                    "-maxrregcount=32",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(parallel=True)},
    options={"build_ext": {"build_lib": "../src/evogp"}},
)

# python setup.py build_ext --inplace