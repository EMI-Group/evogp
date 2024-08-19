import numpy as np
from src.cuda.utils import *

def main():
    top10_t = np.load(f"output/swimmer_0_top10.npy")
    for i in range(10):
        graph_i = to_graph(top10_t[i])
        to_png(graph_i, f"output/swimmer{i}.png")

if __name__ == "__main__":
    main()