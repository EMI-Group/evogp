from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evogp",
    version="0.1.0",
    author="Zhihong Wu, Lishuang Wang, Kebin Sun",
    author_email="zhihong2718@gmail.com",
    description=(
        "Evolutionary Genetic Programming in pure Python, running on evoxc "
        "(EvoX Compiler) with optional compiled evoxc backends."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EMI-Group/evogp",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "scikit_learn",
        "networkx",
        "sympy",
    ],
    extras_require={
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
