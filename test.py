"""
test.py
对给定的 LP 实例执行：
- 读取缓存（若存在）
- 用预训练的图神经网络评估约束重要性
- 对重要约束进行聚合并求解聚合模型
- 使用启发式/子问题方法修复得到的解
- 后处理（邻域搜索等）并记录性能指标

"""

import os
import parser_utils
import utils
import gurobipy as gp
from gurobipy import GRB
import random
import time
from EGAT_models import SpGAT
import torch
import argparse
from torch.autograd import Variable
import repair_and_post_solve_func
import pandas as pd
import numpy as np
from typing import List, Optional
import json
import get_bigraph
import agg_utils

timestamp = time.strftime("%m-%d_%H%M", time.localtime())
print("timestamp:", timestamp)

def calculate_threshold(reduced_costs: List[float], quantile: float = 0.5) -> Optional[float]:
    """
    计算负的 reduced cost 的指定分位数作为阈值。

    参数:
    - reduced_costs: 变量的 reduced cost 列表（可以包含正/负值）
    - quantile: 介于 0 和 1 之间的分位数（例如 0.25 为 25%）

    返回:
    - 若存在负的 reduced cost，返回对应分位数的值；否则返回 None。
    """
    negative_rc = [r for r in reduced_costs if r < 0]
    if not negative_rc:
        # 对于没有负 reduced cost 的情形，返回 None 以便上层逻辑判断
        print("警告：reduced_costs中无负数，无法计算阈值")
        return None

    # numpy.percentile 要求 0..100 的百分比
    threshold = np.percentile(negative_rc, quantile * 100)
    return float(threshold)


def compute_feasibility_metrics(vaule_dict: dict, model: gp.Model, constrs: Optional[List[gp.Constr]] = None):
    """
    计算给定解 `vaule_dict` 在模型约束上的 slack 与违反度统计。

    参数:
    - vaule_dict: 变量名->取值 的字典
    - model: 包含约束的 Gurobi 模型（用于调用 model.getRow）
    - constrs: 可选的约束列表（默认为 model.getConstrs()）

    返回:
    - constr_slack: dict, {idx: slack 或 None}
    - constr_violation: dict, {idx: violation >=0}
    - total_violation: float, 所有约束违反度之和
    - avg_violation: float, 平均违反度（按约束数）
    - num_violated: int, 违反的约束数量
    - max_violation: float, 最大违反度
    """
    if constrs is None:
        constrs = model.getConstrs()

    constr_slack = {}
    constr_violation = {}
    total_violation = 0.0
    max_violation = 0.0
    num_violated = 0
    for idx, c in enumerate(constrs):
        row = model.getRow(c)
        lhs = 0.0
        for j in range(row.size()):
            var = row.getVar(j)
            coeff = row.getCoeff(j)
            lhs += coeff * float(vaule_dict.get(var.VarName, 0.0))

        rhs = float(c.RHS)
        sense = c.Sense
        residual = lhs - rhs

        if sense == '<':
            slack = rhs - lhs
            violation = max(0.0, lhs - rhs)
        elif sense == '>':
            slack = lhs - rhs
            violation = max(0.0, rhs - lhs)
        elif sense == '=':
            slack = None
            violation = abs(residual)
        else:
            slack = None
            violation = abs(residual)

        constr_slack[idx] = float(slack) if slack is not None else None
        constr_violation[idx] = float(violation)
        total_violation += violation
        if violation > 0:
            num_violated += 1
            if violation > max_violation:
                max_violation = violation

    avg_violation = total_violation / len(constrs) if len(constrs) > 0 else 0.0
    return constr_slack, constr_violation, total_violation, avg_violation, num_violated, max_violation

# -----------------------------
# 读取待测试的 LP 实例列表
# - task_name: 指定测试集子目录名
# - lp_dir_path: 存放 .lp 文件的目录
# -----------------------------
# 读问题
# task_name = "CA_500_600_0.5"
# task_name = "CA_750_1100_0.7"
task_name = "IS_1500_6"
lp_dir_path = f"./instance/test/{task_name}"
lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
lp_files.sort()  # 按文件名排序，确保顺序一致
# CHECKPOINT_PATH = f"./model/{task_name}/model_tight_constr.pth"
CHECKPOINT_PATH = f"./model/{task_name}/model_random_sample.pth"

# -----------------------------
# 实验参数
# -----------------------------
if task_name == "CA_750_1100_0.7":
    agg_num = 50
elif task_name == "IS_1500_6":
    agg_num = 300
elif task_name == "CA_500_600_0.5":
    agg_num = 30    
else:
    raise Exception("未定义aggnum")
k0 = 400
k1 = 30
Delta = 30
seed = 3
post_solve_method = "neighborhood"
# post_solve_method = "neighborhood_improved"
# fix_var = False
fix_var = True

repair_method = "subproblem"
if task_name.split("_")[0] == "SC":
    repair_method = "subproblem_cons_geq"
else:
    repair_method = "subproblem"


# -----------------------------
# 加载求解缓存（避免重复完整求解）
# cache 存储着原求解器在不同时间点的中间结果与性能数据
# -----------------------------
cache_dir = "./cache/test"
Threads = 0
solve_num = len(lp_files)
time_limit = 1000
cache_files = os.path.join(cache_dir,task_name+f".json")
cache = utils.load_optimal_cache(cache_files, lp_dir_path, solve_num, Threads,3600)

# -----------------------------
# 解析命令行参数并设置计算设备
# 使用 parser_utils.get_parser("test") 保证训练/测试时网络结构一致
# -----------------------------
parser = parser_utils.get_parser("test")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cpu'
print(f"Using device: {device}")

results = []
result_dir = f"./result/{task_name}_test"
os.makedirs(result_dir,exist_ok=True)

# 在 result_dir 下创建本次运行的配置子文件夹，并将实验配置写入 config.json
config_folder = os.path.join(result_dir, f"config_{timestamp}")
os.makedirs(config_folder, exist_ok=True)

config = {
    "agg_num": agg_num,
    "k0": k0,
    "k1": k1,
    "Delta": Delta,
    "post_solve_method": post_solve_method,
    "fix_var": fix_var,
    "repair_method": repair_method,
    "task_name": task_name,
    "seed": seed,
    "timestamp": timestamp,
}

with open(os.path.join(config_folder, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)


for lp_file in lp_files[:solve_num]:

    # -----------------------------
    # 针对单个 LP 实例的处理主循环
    # - 从缓存读取原始求解信息（若存在）
    # - 构建图表示并通过 GNN 评估约束重要性
    # - 对重要约束聚合并求解聚合模型
    # - 修复并后处理得到的解，记录性能指标
    # -----------------------------

    # 读取问题文件路径
    lp_path = os.path.join(lp_dir_path, lp_file)
    
    # 如果缓存中已有结果，就直接读取，否则求解并写入缓存
    if lp_path in cache:
        print("------------read cache-------------")
        entry = cache[lp_path]
        obj_sense = entry['obj_sense']
        status_orig = entry['status_orig']
        every_second = entry['every_second']
        bks = entry['obj_orig']
        bks_time = entry['time_orig']
        if entry["hit_1000"]:
            # 求解时间超过1000
            gurobi_obj = entry["obj_at_1000"]
            gurobi_time = 1000
        else:
            # 求解时间没超过1000秒
            gurobi_obj = entry['obj_orig']
            gurobi_time = entry['time_orig']
        # obj_orig = entry['obj_orig']
        # time_orig = entry['time_orig']

    else:
        raise Exception("there is not cache")


    # 读取 LP 并记录开始时间
    model_agg = gp.read(lp_path)
    t0 = time.perf_counter()
    numVars = model_agg.getAttr("NumVars")
    cons_num_orig = model_agg.getAttr("NumConstrs")

    # 获取二部图
    features,n,m,edgeA,edgeB,edge_features = get_bigraph.get_bigraph(model=model_agg)

    # 加载网络
    nn_model = SpGAT(nfeat=features.shape[1],  # Feature dimension
                  nhid=args.hidden,  # Feature dimension of each hidden layer
                  nclass=2,  # Number of classes
                  # nclass=int(data_solution[0].max()) + 1,  # Number of classes
                  dropout=args.dropout,  # Dropout
                  nheads=args.nb_heads,  # Number of heads
                  alpha=args.alpha)  # LeakyReLU alpha coefficient
    checkpoint = torch.load(CHECKPOINT_PATH,map_location=device)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    nn_model.eval()


    # 构造网络输入
    features, edgeA, edgeB,edge_features,idx_train = get_bigraph.get_input(nn_model,args,device,n,m,features,edgeA,edgeB,edge_features)

    # 网络前向传播
    output, _ = nn_model(features, edgeA, edgeB,edge_features.detach())

    # 聚合
    sample = agg_utils.get_agg_constr(model_agg,output,idx_train,agg_num)
    # utils.aggregate_constr(model_agg, agg_num, sample)
    utils.aggregate_constr_two_two(model_agg, agg_num, sample)

    cons_num_agg = model_agg.getAttr("NumConstrs")

    print("------------solving agg model-------------")
    vaule_dict,reduced_costs = agg_utils.solve_agg_instance(post_solve_method,model_agg,fix_var,Threads,seed,args)

    # 解的可行性修复
    repair_model,vaule_dict = repair_and_post_solve_func.repair_solution(lp_path,Threads,repair_method,vaule_dict)

    neighborhood = {"k0":k0,"k1":k1,"Delta":Delta}
    bound = 0
    for varname,val in vaule_dict.items():
        bound+= repair_model.getVarByName(varname).Obj * val
    print("bound:",bound)
    if obj_sense == GRB.MINIMIZE:
        repair_model.params.Cutoff = bound + 1e3  # 如果bound距离最优解太近，会导致很难找到可行解。
    else:
        repair_model.params.Cutoff = bound - 1e3 # 如果bound距离最优解太近，会导致很难找到可行解。
    
    repair_and_post_solve_func.PostSolve(repair_model,neighborhood,vaule_dict,lp_file,t0,
                                         time_limit-(time.perf_counter()-t0),post_solve_method,reduced_costs)


    # 指标计算
    utils.summary(repair_model,t0,obj_sense,bks,gurobi_time,every_second,gurobi_obj,results,
            lp_file,status_orig,cons_num_orig,cons_num_agg,bks_time)


df = pd.DataFrame(results)

if post_solve_method == "neighborhood":
    df.to_csv(os.path.join(config_folder, f"result_agg_{agg_num}_fixvar_{fix_var}_{k0}_{k1}_{Delta}_{timestamp}.csv"), index=False)
elif post_solve_method == "neighborhood_improved":
    df.to_csv(os.path.join(config_folder, f"result_agg_{agg_num}_fixvar_{fix_var}_{k0}_{k1}_{Delta}_improved_{timestamp}.csv"), index=False)
else:
    df.to_csv(os.path.join(config_folder, f"result_{post_solve_method}_{timestamp}.csv"), index=False)

