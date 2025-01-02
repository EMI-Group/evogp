import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.environ["MAX_JOBS"] = "8"

setup(
    name="evogp",
    ext_modules=[
        CUDAExtension(
            name="evogp_cuda",
            sources=[
                "./src/evogp/cuda/torch_wrapper.cu",
                "./src/evogp/cuda/generate.cu",
                "./src/evogp/cuda/mutation.cu",
                "./src/evogp/cuda/forward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "--expt-relaxed-constexpr",
                    "--ptxas-options=-v",
                    "-lineinfo" "-lcudart",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(parallel=True)},
    options={
        "build_ext": {
            "build_lib": "./src/evogp"
        }
    }
)
