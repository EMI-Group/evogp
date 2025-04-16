import os
import yaml
from datetime import datetime, timedelta, timezone
import re

# === 配置路径 ===
CONFIG_DIR = "/wuzhihong/evogp2/evogp/experiment/data/configs/"
RUN_SCRIPT = "/wuzhihong/evogp2/evogp/experiment/run4.sh"

# === 固定的实验配置内容 ===
config_dict = {
    "device_name": "4090",
    "seed": 42,
    "pop_size": 100000,
    "generation": 100,
    "problem": {"env_name": "swimmer", "max_episode_length": 100},
    "descriptor": {
        "max_tree_len": 128,
        "max_layer_cnt": 2,
        "using_funcs": ["+", "-", "*", "/"],
        "const_range": [-5, 5],
        "sample_cnt": 100,
        "layer_leaf_prob": 0,
    },
    "crossover": {"name": "LeafBiasedCrossover", "params": {"leaf_bias": 0.95}},
    "mutation": {
        "name": "CombinedMutation",
        "components": [
            {
                "name": "MultiConstMutation",
                "params": {"mutation_rate": 0.8, "mutation_intensity": 0.2},
            },
            {
                "name": "MultiPointMutation",
                "params": {
                    "mutation_rate": 0.8,
                    "mutation_intensity": 0.2,
                    "modify_output": True,
                },
            },
            {"name": "InsertMutation", "params": {"mutation_rate": 0.2}},
        ],
    },
    "selection": {
        "name": "TournamentSelection",
        "params": {
            "tournament_size": 20,
            "best_probability": 0.9,
            "replace": False,
            "survivor_rate": 0.5,
            "elite_rate": 0.05,
        },
    },
}

# === 步骤 1: 生成时间戳文件名 ===
beijing_tz = timezone(timedelta(hours=8))
timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
yaml_filename = f"{timestamp}.yaml"
yaml_path = os.path.join(CONFIG_DIR, yaml_filename)

# === 步骤 2: 写入 YAML 配置文件 ===
os.makedirs(CONFIG_DIR, exist_ok=True)
with open(yaml_path, "w") as f:
    yaml.dump(config_dict, f, sort_keys=False)

print(f"[✓] 配置文件已生成: {yaml_path}")

# === 步骤 3: 更新 run1.sh 中的 --config 路径 ===
if os.path.exists(RUN_SCRIPT):
    with open(RUN_SCRIPT, "r") as f:
        lines = f.readlines()

    updated_lines = []
    modified = False

    for line in lines:
        new_line = re.sub(
            r"(--config\s+experiment/data/configs/)[^/\s]+\.yaml",
            lambda m: m.group(1) + yaml_filename,
            line
        )
        if new_line != line:
            modified = True
        updated_lines.append(new_line)

    if modified:
        with open(RUN_SCRIPT, "w") as f:
            f.writelines(updated_lines)
        print(f"[✓] run1.sh 中的 config 文件名已更新为: {yaml_filename}")
    else:
        print("[!] 未检测到需要替换的路径，run1.sh 未修改")
else:
    print("[!] run1.sh 文件不存在，跳过更新")
