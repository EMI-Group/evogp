from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evogp",
    version="2.0.0a2",
    author="Zhihong Wu, Lishuang Wang, Kebin Sun",
    author_email="zhihong2718@gmail.com",
    description=(
        "Tree-based genetic programming on EvoX Compiler (evoxc) — the torch-free "
        "functional evogp.gp stack, pure Python, no build step."
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
    # The runtime core, and nothing else: `evoxc` (EvoX Compiler) is the
    # evaluation engine every route runs on and `numpy` is the host glue the
    # shipped stack is written in.  No module of `evogp.gp` imports anything
    # heavier, and `import evogp.gp` needs neither torch nor a device.
    install_requires=[
        "evoxc>=0.1.0",
        "numpy",
    ],
    extras_require={
        # `evogp.fused`, the device-native Triton evaluation route: torch at
        # module scope, triton lazily at the first launch.  Needs a CUDA/HIP
        # device and a writable TRITON_CACHE_DIR.
        "fused": ["torch", "triton"],
        # The `iree` backend of the in-graph route (`evogp.gp.GraphEvaluator`).
        "iree": [
            "iree-base-compiler>=3.11.0,<3.12",
            "iree-base-runtime>=3.11.0,<3.12",
        ],
        # `evogp.evox_ext`, the EvoX 2.0 `StdWorkflow` adapter — the only
        # subtree that imports `evox`.
        "evox": ["evox>=2.0.0a1"],
        "dev": ["pytest"],
        # `example/gp_uci_sr.py` symbolic-regresses scikit-learn's bundled
        # `load_diabetes` table; the library itself does not import it.
        "examples": ["scikit-learn"],
    },
    include_package_data=True,
    # The wheel ships the package `.py` modules PLUS the in-package Markdown the
    # module docstrings and the reports point at by path: `evogp/CONTEXT.md` at
    # the top package's root, `evogp/gp/DESIGN.md` one level down and the same
    # in the other two nested packages.  Declared here rather than through
    # MANIFEST.in so the wheel never depends on the sdist rules.  The patterns
    # are flat (`*` never crosses a `/`), so `evogp/*/tests/CONTEXT.md` stays
    # out -- `tests/` is not a package and its CONTEXT.md is dev material --
    # and `evogp/gp/evidence/*.json` is sdist-only (`exclude_package_data`).
    package_data={
        "evogp": ["*.md", "*/*.md"],
        "evogp.gp": ["*.md"],
        "evogp.fused": ["*.md"],
        "evogp.evox_ext": ["*.md"],
    },
    exclude_package_data={
        "evogp.gp": ["evidence/*", "evidence/*/*"],
    },
    python_requires=">=3.11",
)
