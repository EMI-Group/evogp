import numpy as np
from src.cuda.utils import *

def main():
    top10_t = np.load(f"output/output16/halfcheetah_0_top10.npy")
    for i in [9]:
        graph_i = from_cuda_node(top10_t[i])
        print(graph_i)

if __name__ == "__main__":
    main()