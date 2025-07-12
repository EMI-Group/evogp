import os
import csv
import yaml
import time
import shutil
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path

from evogp.tree import Forest, GenerateDescriptor
from evogp.problem import BraxProblem
from evogp.algorithm import (
    GeneticProgramming,
    TournamentSelection,
    DefaultSelection,
    CombinedMutation,
    MultiConstMutation,
    MultiPointMutation,
    InsertMutation,
    DeleteMutation,
    DefaultMutation,
    LeafBiasedCrossover,
)

# === 构造器 ===

def build_problem(cfg):
    return BraxProblem(cfg["env_name"], cfg["max_episode_length"])

def build_descriptor(cfg, input_len, output_len):
    return GenerateDescriptor(
        max_tree_len=cfg["max_tree_len"],
        input_len=input_len,
        output_len=output_len,
        using_funcs=cfg["using_funcs"],
        max_layer_cnt=cfg["max_layer_cnt"],
        const_range=cfg["const_range"],
        sample_cnt=cfg["sample_cnt"],
        layer_leaf_prob=cfg["layer_leaf_prob"]
    )

def build_crossover(cfg):
    if cfg["name"] == "LeafBiasedCrossover":
        return LeafBiasedCrossover(**cfg["params"])
    else:
        raise NotImplementedError(f"Unknown crossover: {cfg['name']}")
    
def build_single_mutation(name, params, descriptor):
    if name == "MultiConstMutation":
        return MultiConstMutation(descriptor=descriptor, **params)
    elif name == "MultiPointMutation":
        return MultiPointMutation(descriptor=descriptor, **params)
    elif name == "InsertMutation":
        return InsertMutation(descriptor=descriptor, **params)
    elif name == "DeleteMutation":
        return DeleteMutation(**params)
    elif name == "DefaultMutation":
        return DefaultMutation(descriptor=descriptor, **params)
    else:
        raise NotImplementedError(f"Unknown mutation type: {name}")
    
def build_mutation(cfg, descriptor):
    if cfg["name"] == "CombinedMutation":
        components = []
        for comp in cfg["components"]:
            name = comp["name"]
            params = comp["params"]
            components.append(build_single_mutation(name, params, descriptor))
        return CombinedMutation(components)
    else:
        return build_single_mutation(cfg["name"], cfg["params"], descriptor)

def build_selection(cfg):
    if cfg["name"] == "TournamentSelection":
        return TournamentSelection(**cfg["params"])
    elif cfg["name"] == "DefaultSelection":
        return DefaultSelection(**cfg["params"])
    else:
        raise NotImplementedError(f"Unknown selection: {cfg['name']}")

# === 实验主逻辑 ===

def run_experiment(config_path):
    # 用 YAML 文件名作为 ID
    config_id = Path(config_path).stem
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])

    # 初始化问题和算法
    problem = build_problem(config["problem"])
    descriptor = build_descriptor(config["descriptor"], problem.problem_dim, problem.solution_dim)
    crossover = build_crossover(config["crossover"])
    mutation = build_mutation(config["mutation"], descriptor)
    selection = build_selection(config["selection"])
    algorithm = GeneticProgramming(
        initial_forest=Forest.random_generate(pop_size=config["pop_size"], descriptor=descriptor),
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        enable_pareto_front=True,
    )

    # 日志记录初始化
    os.makedirs("experiment/data/results", exist_ok=True)
    os.makedirs("experiment/data/configs", exist_ok=True)
    result_path = f"experiment/data/results/{config_id}11.csv"
    log_headers = [
        "generation", "best_fitness", "mean_fitness",
        "max_tree_size", "mean_tree_size", "min_tree_size",
        "eval_time", "alg_time", "total_time"
    ]
    with open(result_path, "w", newline="") as f:
        csv.writer(f).writerow(log_headers)

    # 开始实验
    total_time = 0
    for i in range(config["generation"]):
        t0 = time.time()
        fitness = problem.evaluate(algorithm.forest)
        torch.cuda.synchronize()
        eval_time = time.time() - t0

        t1 = time.time()
        algorithm.step(fitness)
        torch.cuda.synchronize()
        alg_time = time.time() - t1

        total_time += time.time() - t0
        scores = fitness.cpu().numpy()
        valid = scores[~np.isinf(scores)]
        size = algorithm.forest.batch_subtree_size[:, 0].cpu().numpy()

        row = [
            i, max(valid), np.mean(valid),
            max(size), np.mean(size), min(size),
            eval_time, alg_time, total_time
        ]
        print(row)
        with open(result_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # 保存模型
    with open(f"experiment/data/results/{config_id}.pkl", "wb") as f:
        pickle.dump(algorithm.pareto_front, f)

    # 保存 py 脚本副本
    shutil.copy(Path(__file__), f"experiment/data/configs/{config_id}.py")

    print(f"[✓] Experiment {config_id} complete.")

# === 命令行入口 ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiment/data/configs/test_halfcheetah.yaml")
    args = parser.parse_args()
    run_experiment(args.config)