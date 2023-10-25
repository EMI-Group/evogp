mkdir build
pybind_include_path=$(python3 -c "import pybind11; print(pybind11.get_include())")
python_executable=$(python3 -c 'import sys; print(sys.executable)')
nvcc --threads 4 -Xcompiler -Wall -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -DTEST -Xcompiler -O3 --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] --generate-code=arch=compute_89,code=[compute_89,sm_89] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -x cu -c kernel.cu -o build/kernel.cu.o
c++ -I/usr/local/cuda/include -I$pybind_include_path $(${python_executable}-config --cflags) -O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects -o build/gpu_ops.cpp.o -c gpu_ops.cpp
c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared  -o build/gpu_ops$(${python_executable}-config --extension-suffix) build/gpu_ops.cpp.o build/kernel.cu.o -L/usr/local/cuda/lib64  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
strip build/gpu_ops$(${python_executable}-config --extension-suffix)