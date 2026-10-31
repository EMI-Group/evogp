from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evogp",
    version="2.0.0a1",
    author="Zhihong Wu, Lishuang Wang, Kebin Sun",
    author_email="zhihong2718@gmail.com",
    description=(
        "Tree-based genetic programming on EvoX Compiler (evoxc) — the EvoGP "
        "release line aligned with EvoX 2.0.0a1, pure Python, no build step."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EMI-Group/evogp",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # The runtime core: `evoxc` (EvoX Compiler) is the evaluation engine every
    # route runs on, and `evox` 2.0.0a1 is the functional framework this release
    # line aligns with.  Neither is imported by `evogp.tree` at module scope, so
    # the pure-Python / no-build-step property is unaffected.
    # `torch` is deliberately optional (`evogp[torch]`): the tensor-native stack
    # in `evogp.tree`/`algorithm`/`problem`/`pipeline` needs it, while the
    # functional stack in `evogp.gp` runs on `evoxc` + numpy alone.
    install_requires=[
        "evoxc>=0.1.0",
        "evox>=2.0.0a1",
        "numpy",
        "scikit_learn",
        "networkx",
        "sympy",
    ],
    extras_require={
        # The tensor-native GP stack and its benchmark/test oracles.
        "torch": ["torch"],
        # The fused Triton evaluation route (`EVOGP_EVAL_MODE=fused`): a Triton
        # JIT plus a CUDA-visible device (AMD GPUs under ROCm included).
        "fused": ["torch", "triton"],
        "visualize": ["pygraphviz"],
        "brax": ["jax", "brax"],
        "iree": [
            "iree-base-compiler>=3.11.0,<3.12",
            "iree-base-runtime>=3.11.0,<3.12",
        ],
        "dev": ["pytest"],
    },
    include_package_data=True,
    python_requires=">=3.11",
)
