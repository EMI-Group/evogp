import numpy as np
from src.cuda.utils import *

def main():
    top10_t = np.load(f"output/output16/halfcheetah_0_top10.npy")
    for i in [3]:
        graph_i = from_cuda_node(top10_t[i])
        print(to_string(graph_i[1], graph_i[0]))
        print(to_infix(graph_i[2][0], graph_i[1], graph_i[0]))
        print(graph_i)

if __name__ == "__main__":
    main()