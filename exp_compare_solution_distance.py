import time
import csv
import gurobipy
from gurobipy import GRB
import argparse
import random
import os
import numpy as np
import torch
from predictandsearch.helper import get_a_new2
import json
from typing import List, Optional
import parser_utils
import utils
import gurobipy as gp
from gurobipy import GRB
from EGAT_models import SpGAT
import argparse
from torch.autograd import Variable
import pandas as pd

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


def compute_manhattan_distance(true_sol: dict, pred_sol: dict) -> float:
    """
    计算两个解字典之间的曼哈顿距离（按键匹配）。

    参数:
    - true_sol: 真实解字典，键为变量名，值为数值
    - pred_sol: 预测解字典，键为变量名，值为数值

    返回:
    - 两个字典在共有键上的曼哈顿距离（sum |pred - true|）
    """
    total = 0.0
    if not isinstance(true_sol, dict) or not isinstance(pred_sol, dict):
        return total
    for key, true_val in true_sol.items():
        pred_val = pred_sol.get(key)
        if pred_val is None:
            continue
        try:
            total += abs(float(pred_val) - float(true_val))
        except Exception:
            # 忽略无法转换为 float 的值
            continue
    return total



def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1, delta
    '''
    if task=="IP":
        return 400,5,1
    elif task == "IS":
        return 300,300,15
    elif task == "WA":
        return 0,600,5
    elif task == "CA_500_600":
        return 400,0,10
    elif task == "CA_750_1100_0.7":
        return 400,30,100
        # return 1,1,2
    elif task == "IS_1500_6":
        return 400,30,100
    else:
        raise NotImplementedError(f"task {task} not implemented")

timestamp = time.strftime("%m-%d_%H%M", time.localtime())
print("timestamp:", timestamp)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# 统一默认参数（可按需修改）
# DEFAULT_TASK_NAME = 'CA_750_1100_0.7'
DEFAULT_TASK_NAME = 'IS_1500_6'
DEFAULT_TEST_NUM = 10
DEFAULT_K0, DEFAULT_K1, DEFAULT_DELTA = test_hyperparam(DEFAULT_TASK_NAME)

result_dir = f"./result/{DEFAULT_TASK_NAME}_test"
os.makedirs(result_dir,exist_ok=True)

# 在 result_dir 下创建本次运行的配置子文件夹，并将实验配置写入 config.json
config_folder = os.path.join(result_dir, f"config_{timestamp}_primal_solution_analysis")
os.makedirs(config_folder, exist_ok=True)


def compute_ps_manhattan_distance(TaskName, TestNum, k_0, k_1, delta):
    project_dir = '/home/cc/code/modeling_strategy/predictandsearch'


    #set log folder
    solver='GRB'
    # test_task = f'{TaskName}_{solver}_Predect&Search'

    # load pretrained model
    if TaskName == "IP":
        from predictandsearch.GCN import GNNPolicy_position as GNNPolicy, postion_get
    else:
        from predictandsearch.GCN import GNNPolicy

    model_name = f'{TaskName}.pth'
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    pathstr = os.path.join(models_dir, model_name)
    policy = GNNPolicy().to(DEVICE)
    state = torch.load(pathstr, map_location=torch.device('cuda:0'))
    policy.load_state_dict(state)

    sample_names = sorted(os.listdir(f'./instance/test/{TaskName}'))

    # load cache JSON
    cache_path = os.path.join(os.path.dirname(project_dir), 'cache', 'test', f'{TaskName}.json')
    cache_data = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as jf:
                cache_data = json.load(jf)
        except Exception:
            cache_data = {}

    # result_dir = os.path.join(project_dir, 'result', f'{TaskName}_test')
    # os.makedirs(result_dir, exist_ok=True)

    results = {}
    ps_var_lists = {}

    for ins_num in range(TestNum):
        t1 = time.perf_counter()
        test_ins_name = sample_names[ins_num]
        ins_name_to_read = f'./instance/test/{TaskName}/{test_ins_name}'

        # get bipartite graph as input
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
        constraint_features = c_nodes.cpu()
        constraint_features[np.isnan(constraint_features)] = 1
        variable_features = v_nodes
        if TaskName == "IP":
            variable_features = postion_get(variable_features)
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        # prediction
        BD = policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
        ).sigmoid().cpu().squeeze()

        # align variable names
        all_varname = [name for name in v_map]
        binary_name = [all_varname[i] for i in b_vars]
        scores = []
        for i in range(len(v_map)):
            ttype = "C"
            if all_varname[i] in binary_name:
                ttype = 'BINARY'
            scores.append([i, all_varname[i], BD[i].item(), -1, ttype])

        scores.sort(key=lambda x: x[2], reverse=True)

        binary_scores = [[s[0], s[1], s[2]] for s in scores if s[4] == 'BINARY']
        binary_scores.sort(key=lambda x: x[2], reverse=True)

        ps_fix_solution = {item[1]: (1 if item[2] >= 0.5 else 0) for item in binary_scores}


        # compute Manhattan with cached solution
        key = f"./instance/test/{TaskName}/{test_ins_name}"
        entry = cache_data.get(key)
        manhattan = None
        bks = None
        bks_time = None
        if entry is not None:
            sol = entry.get('solution', {})
            bks = entry.get('obj_orig')
            bks_time = entry.get('time_orig')
            manhattan = compute_manhattan_distance(sol, ps_fix_solution)

        # compute violations using ps_fix_solution
        obj_sense = None
        try:
            model_ps = gurobipy.read(ins_name_to_read)
            # record objective sense for debugging (minimize / maximize)
            try:
                if model_ps.ModelSense == GRB.MINIMIZE:
                    obj_sense = 'minimize'
                elif model_ps.ModelSense == GRB.MAXIMIZE:
                    obj_sense = 'maximize'
                else:
                    obj_sense = str(model_ps.ModelSense)
            except Exception:
                obj_sense = None
            cons = model_ps.getConstrs()
            vars_all = model_ps.getVars()
            num_constraints = len(cons)
            num_variables = len(vars_all)
            violations = {}
            for ci, c in enumerate(cons):
                row = model_ps.getRow(c)
                lhs = 0.0
                for j in range(row.size()):
                    var = row.getVar(j)
                    coef = row.getCoeff(j)
                    varname = var.VarName
                    xpred = ps_fix_solution.get(varname, 0)
                    lhs += coef * xpred
                rhs = c.RHS
                sense = c.Sense
                if sense == '<':
                    viol = max(0.0, lhs - rhs)
                elif sense == '>':
                    viol = max(0.0, rhs - lhs)
                elif sense == '=':
                    viol = abs(lhs - rhs)
                else:
                    viol = abs(lhs - rhs)
                violations[ci] = float(viol)
            violation_sum = sum(violations.values()) if violations else 0.0
            violation_avg = violation_sum / num_constraints if num_constraints > 0 else None
        except Exception:
            violations = {}
            num_constraints = None
            num_variables = None
            violation_sum = None
            violation_avg = None
            model_ps = None

        current_obj = 0
        for var in vars_all:
            varname = var.VarName
            xpred = ps_fix_solution.get(varname, 0)
            coef = var.Obj
            current_obj += coef * xpred

        try:
            if bks is None or bks == 0 or current_obj is None:
                obj_gap_percent = None
            else:
                obj_gap_percent = 100.0 * abs(current_obj - bks) / abs(bks)
        except Exception:
            obj_gap_percent = None

        # partial solutions: top k1 -> 1, bottom k0 -> 0
        top_k1_list = [item[1] for item in binary_scores[:k_1]] if k_1 > 0 else [] # 这里不能直接取k_1个，还要加上判断概率是否大于0.5
        asc_binary_scores = sorted(binary_scores, key=lambda x: x[2])
        bottom_k0_list = [item[1] for item in asc_binary_scores[:k_0]] if k_0 > 0 else []

        pred_partial_top = {n: 1 for n in top_k1_list}
        pred_partial_bottom = {n: 0 for n in bottom_k0_list}

        # combined partial: apply both top_k1=1 and bottom_k0=0
        pred_partial_both = pred_partial_top.copy()
        for n in pred_partial_bottom:
            # bottom set to 0 overrides any previous
            pred_partial_both[n] = 0

        manhattan_partial_top = compute_manhattan_distance(sol, pred_partial_top)
        manhattan_partial_bottom = compute_manhattan_distance(sol, pred_partial_bottom)
        manhattan_partial_both = compute_manhattan_distance(sol, pred_partial_both)

        ps_var_lists[test_ins_name] = {
            'top_k1': top_k1_list,
            'bottom_k0': bottom_k0_list,
        }

        # 统计 PS 完整解的 0/1 比例以及真实解的 0/1 比例
        ps_one_count = sum(1 for v in ps_fix_solution.values() if v == 1)
        ps_zero_count = sum(1 for v in ps_fix_solution.values() if v == 0)
        total_ps_vars = ps_one_count + ps_zero_count if (ps_one_count + ps_zero_count) > 0 else None
        ps_one_pct = (ps_one_count / total_ps_vars) if total_ps_vars else None
        ps_zero_pct = (ps_zero_count / total_ps_vars) if total_ps_vars else None

        true_one_count = sum(1 for v in sol.values() if float(v) >= 0.5) if sol else None
        true_zero_count = sum(1 for v in sol.values() if float(v) < 0.5) if sol else None
        total_true_vars = true_one_count + true_zero_count if (true_one_count is not None and true_zero_count is not None) else None
        true_one_pct = (true_one_count / total_true_vars) if total_true_vars else None
        true_zero_pct = (true_zero_count / total_true_vars) if total_true_vars else None

        results[test_ins_name] = {
            'manhattan': manhattan,
            'manhattan_partial_top_k1': manhattan_partial_top,
            'manhattan_partial_bottom_k0': manhattan_partial_bottom,
            'manhattan_partial_both_k0k1': manhattan_partial_both,
            'top_k1_list': top_k1_list,
            'bottom_k0_list': bottom_k0_list,
            'obj':current_obj,
            'obj_gap_percent': obj_gap_percent,
            'bks': bks,
            'bks_time': bks_time,
            'violation_sum': violation_sum,
            'violation_avg': violation_avg,
            'num_constraints': num_constraints,
            'num_variables': num_variables,
            'obj_sense': obj_sense,
            'violations': violations,
            'ps_one_count': ps_one_count,
            'ps_zero_count': ps_zero_count,
            'ps_one_pct': ps_one_pct,
            'ps_zero_pct': ps_zero_pct,
            'true_one_count': true_one_count,
            'true_zero_count': true_zero_count,
            'true_one_pct': true_one_pct,
            'true_zero_pct': true_zero_pct,
            # PS partial counts (按总变量数归一化)
            'ps_partial_top_one_count': sum(1 for n in top_k1_list if ps_fix_solution.get(n, 0) == 1),
            'ps_partial_top_one_pct': (sum(1 for n in top_k1_list if ps_fix_solution.get(n, 0) == 1) / total_ps_vars) if total_ps_vars else None,
            'ps_partial_bottom_zero_count': sum(1 for n in bottom_k0_list if ps_fix_solution.get(n, 0) == 0),
            'ps_partial_bottom_zero_pct': (sum(1 for n in bottom_k0_list if ps_fix_solution.get(n, 0) == 0) / total_ps_vars) if total_ps_vars else None,
            'ps_partial_both_one_count': sum(1 for n,v in pred_partial_both.items() if v == 1),
            'ps_partial_both_one_pct': (sum(1 for n,v in pred_partial_both.items() if v == 1) / total_ps_vars) if total_ps_vars else None,
            # true partial counts (按总变量数归一化)
            'true_partial_top_one_count': sum(1 for n in top_k1_list if float(sol.get(n, 0)) >= 0.5) if sol else None,
            'true_partial_top_one_pct': (sum(1 for n in top_k1_list if float(sol.get(n, 0)) >= 0.5) / total_ps_vars) if (sol and total_ps_vars) else None,
            'true_partial_bottom_one_count': sum(1 for n in bottom_k0_list if float(sol.get(n, 0)) >= 0.5) if sol else None,
            'true_partial_bottom_one_pct': (sum(1 for n in bottom_k0_list if float(sol.get(n, 0)) >= 0.5) / total_ps_vars) if (sol and total_ps_vars) else None,
            'true_partial_both_one_count': sum(1 for n,v in pred_partial_both.items() if float(sol.get(n, 0)) >= 0.5) if sol else None,
            'true_partial_both_one_pct': (sum(1 for n,v in pred_partial_both.items() if float(sol.get(n, 0)) >= 0.5) / total_ps_vars) if (sol and total_ps_vars) else None,
        }

    out_path = os.path.join(config_folder, f'manhattan_results_{TaskName}_ps.json')
    with open(out_path, 'w', encoding='utf-8') as of:
        json.dump(results, of, indent=2)
    return results, ps_var_lists




 
def compute_sc_manhattan_distance(
    task_name: str = DEFAULT_TASK_NAME,
    agg_num: int = 50,
    k0: int = DEFAULT_K0,
    k1: int = DEFAULT_K1,
    Delta: int = 100,
    seed: int = 3,
    # post_solve_method: str = "neighborhood_improved",
    post_solve_method: str = "neighborhood",
    ps_var_lists: dict = None,
):
    # -----------------------------
    # 读取待测试的 LP 实例列表
    # - task_name: 指定测试集子目录名
    # - lp_dir_path: 存放 .lp 文件的目录
    # -----------------------------
    # 读问题
    lp_dir_path = f"./instance/test/{task_name}"
    lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致


    post_solve_method = "neighborhood"
    # post_solve_method = "neighborhood_improved"
    # post_solve_method = "fix"
    # post_solve_method = "warm_start"
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

    results = {}

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

        # obj sense
        obj_type = model_agg.ModelSense
        if obj_type == GRB.MINIMIZE:
            obj_type = 'minimize'
        elif obj_type == GRB.MAXIMIZE:
            obj_type = 'maximize'
        else:
            raise Exception("unknown obj sense")

        # -----------------------------
        # 提取变量（columns）信息：目标系数、上下界、类型等
        # -----------------------------
        vars = model_agg.getVars()
        n = len(vars)
        coefficient = [v.obj for v in vars]
        lower_bound = [v.lb for v in vars]
        upper_bound = [v.ub for v in vars]
        value_type = [{'B': 'B', 'I': 'I', 'C': 'C'}.get(v.vtype, 'C') for v in vars]

        # -----------------------------
        # 提取约束（rows）信息：系数位置、RHS、约束类型、行的度（非零个数）等
        # -----------------------------
        constrs = model_agg.getConstrs()
        m = len(constrs)
        sense_map = {'<': 1, '>': 2, '=': 3}
        k, site, value, constraint, constraint_type = [], [], [], [], []
        constr_degree = []
        variable_degree = [0 for i in range(n)]
        for c in constrs:
            row = model_agg.getRow(c)
            vars_in_row = [row.getVar(idx) for idx in range(row.size())]
            coeffs = [row.getCoeff(idx) for idx in range(row.size())]

            k.append(len(vars_in_row))
            site.append([v.index for v in vars_in_row])  # 变量下标
            value.append([float(co) for co in coeffs])
            constraint.append(c.RHS)
            constraint_type.append(sense_map[c.Sense])
            constr_degree.append(row.size())  # 度
            for idx in range(row.size()):
                var = row.getVar(idx)
                variable_degree[var.index] += 1

        norm_variable_degree = utils.z_score_normalize(variable_degree)
        norm_constr_degree = utils.z_score_normalize(constr_degree)

        # -----------------------------
        # 将线性规划问题编码为二部图（变量节点 + 约束节点）
        # - variable_features / constraint_features: 节点特征
        # - edge_indices / edge_features: 边与边特征（系数）
        # -----------------------------
        variable_features = []
        constraint_features = []
        edge_indices = [[], []]
        edge_features = []

        # print(value_type)
        norn_coeff = utils.z_score_normalize(coefficient)
        for i in range(n):
            now_variable_features = []
            now_variable_features.append(norn_coeff[i])
            now_variable_features.append(0)  #
            now_variable_features.append(1)  # [0,1]代表变量
            if (value_type[i] == 'C'):
                now_variable_features.append(0)
            else:
                now_variable_features.append(1)
            now_variable_features.append(random.random())

            # 度
            now_variable_features.append(norm_variable_degree[i])

            variable_features.append(now_variable_features)


        for i in range(m):
            now_constraint_features = []
            now_constraint_features.append(constraint[i])
            if (constraint_type[i] == 1):
                now_constraint_features.append(1)
                now_constraint_features.append(0)
                now_constraint_features.append(0)
            if (constraint_type[i] == 2):
                now_constraint_features.append(0)
                now_constraint_features.append(1)
                now_constraint_features.append(0)
            if (constraint_type[i] == 3):
                now_constraint_features.append(0)
                now_constraint_features.append(0)
                now_constraint_features.append(1)
            now_constraint_features.append(random.random())

            # 度
            now_constraint_features.append(norm_constr_degree[i])

            # pos_emb
            # pos_emb = utils.decimal_to_binary_list(m, i)
            # now_constraint_features.extend(pos_emb)

            constraint_features.append(now_constraint_features)

        for i in range(m):
            for j in range(k[i]):
                edge_indices[0].append(i)
                edge_indices[1].append(site[i][j])
                edge_features.append([value[i][j]])

        # change
        n = len(variable_features)
        var_size = len(variable_features[0])
        m = len(constraint_features)
        con_size = len(constraint_features[0])

        edge_num = len(edge_indices[0])

        edgeA = []
        edgeB = []
        # edge_features = []
        for i in range(edge_num):
            edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
            edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])

        edgeA = torch.as_tensor(edgeA)
        edgeB = torch.as_tensor(edgeB)
        edge_features = torch.as_tensor(edge_features)

        if var_size > con_size:
            for i in range(m):
                for j in range(var_size - con_size):
                    constraint_features[i].append(0)
        else:
            for i in range(n):
                for j in range(con_size-var_size):
                    variable_features[i].append(0)


        features = variable_features + constraint_features
        features = torch.as_tensor(features).float()


        # focal loss的权重
        # num_label = [2, 1]
        # num_label = [1, 1]
        # num_label = torch.as_tensor(num_label).to(device)

        # 需要计算loss的部分，也就是约束的范围
        idx_train = torch.as_tensor(range(n, n + m))


        ## 过网络
        # 读取网络
        nn_model = SpGAT(nfeat=features.shape[1],  # Feature dimension
                    nhid=args.hidden,  # Feature dimension of each hidden layer
                    nclass=2,  # Number of classes
                    # nclass=int(data_solution[0].max()) + 1,  # Number of classes
                    dropout=args.dropout,  # Dropout
                    nheads=args.nb_heads,  # Number of heads
                    alpha=args.alpha)  # LeakyReLU alpha coefficient

        # todo:模型重新改一下
        CHECKPOINT_PATH = f"./model/{task_name}/model_tight_constr.pth"
        # CHECKPOINT_PATH = f"./model/{task_name}/model_random_sample.pth"
        checkpoint = torch.load(CHECKPOINT_PATH,map_location=device)
        nn_model.load_state_dict(checkpoint['model_state_dict'])
        nn_model.eval()
        if args.cuda:  # Move to GPU
            nn_model.to(device)

            features = features.to(device)

            edgeA = edgeA.to(device)
            edgeB = edgeB.to(device)
            edge_features = edge_features.to(device)
            idx_train = idx_train.to(device)


        features = Variable(features)
        edgeA = Variable(edgeA)
        edgeB = Variable(edgeB)
        # Define computation graph for automatic differentiation
        output, _ = nn_model(features, edgeA, edgeB,edge_features.detach())
        time_network_finish = time.perf_counter()
        time_network = time_network_finish - t0


        ## 聚合
        # 得到需要聚合的约束
        constr_score = output[idx_train].detach().numpy().tolist()
        constr_idx_score =  [[idx,item[1]]for idx,item in enumerate(constr_score)]
        constr_idx_score.sort(key=lambda x:x[1], reverse=True)

        agg_constr_idx = [constr_idx_score[i][0] for i in range(agg_num)]

        # 聚合求解
        conss = model_agg.getConstrs()
        sample = [conss[constr_idx] for constr_idx in agg_constr_idx]
        # utils.aggregate_constr(model_agg, agg_num, sample)
        utils.aggregate_constr_two_two(model_agg, agg_num, sample)

        cons_num_agg = model_agg.getAttr("NumConstrs")
        print("------------solving agg model-------------")

        # get reduced cost
        reduced_costs = None
        if post_solve_method == "neighborhood_improved":
            model_agg_relax = model_agg.copy()
            model_agg_relax = model_agg_relax.relax()
            model_agg_relax.optimize()
            reduced_costs = [v.RC for v in model_agg_relax.getVars()]

        if fix_var:
            # 备份，用于固定变量后的求解
            model_agg_2 = model_agg.copy()
            agg_model_solve_time = args.agg_model_solve_time
            # 聚合求解
            model_agg_2.setParam("Threads",Threads)
            model_agg_2.setParam("Seed", seed+1)
            model_agg_2.setParam("TimeLimit", agg_model_solve_time)
            model_agg_2.optimize()

            # 获得聚合问题的解（满足整数性）
            Vars = model_agg_2.getVars()
            vaule_dict = {var.VarName: var.X for var in Vars}

            ##  松弛
            for v in model_agg.getVars():
                try:
                    v.VType = GRB.CONTINUOUS
                except Exception:
                    # 如果直接赋值失败，忽略（继续）
                    pass
            model_agg.update()

            # 求解松弛聚合问题
            model_agg.setParam("Threads",Threads)
            model_agg.setParam("Seed", seed)
            if agg_model_solve_time == -1:
                model_agg.optimize()
            else:
                model_agg.setParam("TimeLimit", agg_model_solve_time)
                model_agg.optimize()

            # 获得lp松弛解
            # Vars = model_agg.getVars()
            # vaule_dict_lp_relax = {var.VarName: var.X for var in Vars}


            dual_values = [c.Pi for c in model_agg.getConstrs()]
            reduced_costs = [v.RC for v in model_agg.getVars()]
            if sum([1 if r < 0 else 0 for r in reduced_costs]) == 0:
                print("不固定")
            else:
                threshold = sum([r if r < 0 else 0 for r in reduced_costs]) / sum([1 if r < 0 else 0 for r in reduced_costs])
                # threshold = calculate_threshold(reduced_costs,0.25)
                fixed_vars = [v.VarName for v in model_agg.getVars() if v.RC <= threshold]

                cnt = 0
                for varname in fixed_vars:
                    if vaule_dict[varname] != 0:
                        cnt+=1
                    vaule_dict[varname] = 0
                print(f"额外固定了：{cnt}")


            time_solve_agg_model_finish = time.perf_counter()
            time_solve_agg_model = time_solve_agg_model_finish - time_network_finish # 这里可能不太对，这里不只是聚合了，还有包括一个子问题求解的过程。

        else:
            model_agg.setParam("Threads", Threads)
            agg_model_solve_time = args.agg_model_solve_time
            if agg_model_solve_time == -1:
                model_agg.optimize()
            else:
                model_agg.setParam("TimeLimit", agg_model_solve_time)
                model_agg.optimize()
            time_solve_agg_model_finish = time.perf_counter()
            time_solve_agg_model = time_solve_agg_model_finish - time_network_finish

            agg_objval_original = model_agg.ObjVal

            # 获得变量值
            Vars = model_agg.getVars()
            vaule_dict = {var.VarName: var.X for var in Vars}


        # compute Manhattan (Hamming for binaries) distance between obtained solution
        # and cached (true) solution. Binarize fractional values using 0.5 threshold.
        bks_solution = cache.get(lp_path, {}).get('solution', {})
        count = 0

        # 初始化违反度相关变量
        violations = {}
        num_constraints = None
        num_variables = None
        violation_sum = None
        violation_avg = None

        if bks_solution:
            count = compute_manhattan_distance(bks_solution, vaule_dict)

        # --- 计算每个约束的违反度（使用二值化的预测解） ---
        try:
            cons = model_agg.getConstrs()
            vars_all = model_agg.getVars()
            num_constraints = len(cons)
            num_variables = len(vars_all)
            violations = {}
            for ci, c in enumerate(cons):
                row = model_agg.getRow(c)
                lhs = 0.0
                for j in range(row.size()):
                    var = row.getVar(j)
                    coef = row.getCoeff(j)
                    varname = var.VarName
                    # 使用二值化预测值评估违反度
                    val = vaule_dict.get(varname, 0)
                    lhs += coef * val
                rhs = c.RHS
                sense = c.Sense
                if sense == '<':
                    viol = max(0.0, lhs - rhs)
                elif sense == '>':
                    viol = max(0.0, rhs - lhs)
                elif sense == '=':
                    viol = abs(lhs - rhs)
                else:
                    viol = abs(lhs - rhs)
                violations[ci] = float(viol)
            violation_sum = sum(violations.values()) if violations else 0.0
            violation_avg = violation_sum / num_constraints if num_constraints > 0 else None
        except Exception:
            violations = {}
            num_constraints = None
            num_variables = None
            violation_sum = None
            violation_avg = None

        # 计算当前解的目标函数值并与缓存中的最优值计算 gap
        try:
            Vars_for_obj = model_agg.getVars()
            current_obj = 0.0
            for v in Vars_for_obj:
                current_obj += float(v.obj) * float(vaule_dict.get(v.VarName, 0.0))
        except Exception:
            current_obj = None

        # 计算相对 gap（百分比），使用缓存中的 obj_orig (bks)
        try:
            if bks is None or bks == 0 or current_obj is None:
                obj_gap_percent = None
            else:
                obj_gap_percent = 100.0 * abs(current_obj - bks) / abs(bks)
        except Exception:
            obj_gap_percent = None

        # 计算部分解曼哈顿距离（如果提供了 PS 的变量列表）
        manhattan_partial_top = None
        manhattan_partial_bottom = None
        if ps_var_lists is not None:
            # ps_var_lists 的键使用文件名（与 lp_file 一致）
            lists = ps_var_lists.get(lp_file)
            if lists is None:
                # try without extension fallback
                lists = ps_var_lists.get(os.path.basename(lp_file))
            if lists is not None:
                top_k1_list = lists.get('top_k1', [])
                bottom_k0_list = lists.get('bottom_k0', [])
                # 使用 SC 的解值（vaule_dict）作为部分解的值，变量集合来自 PS 的 top/bottom 列表
                pred_partial_top = {n: vaule_dict.get(n, 0) for n in top_k1_list}
                pred_partial_bottom = {n: vaule_dict.get(n, 0) for n in bottom_k0_list}
                bks_solution = cache.get(lp_path, {}).get('solution', {})
                manhattan_partial_top = compute_manhattan_distance(bks_solution, pred_partial_top)
                manhattan_partial_bottom = compute_manhattan_distance(bks_solution, pred_partial_bottom)
                # combined partial: 合并 top 与 bottom 的变量集合，值均来自 vaule_dict（bottom 可覆盖 top）
                pred_partial_both = {n: vaule_dict.get(n, 0) for n in set(top_k1_list + bottom_k0_list)}
                # 如果 bottom_k0_list 中的变量需要特别处理（例如强制为 0），可在此覆盖
                manhattan_partial_both = compute_manhattan_distance(bks_solution, pred_partial_both)

        # 保存每实例的距离、目标与违反度信息
        results[lp_file] = {
            'manhattan': count,
            'manhattan_partial_top_k1': manhattan_partial_top,
            'manhattan_partial_bottom_k0': manhattan_partial_bottom,
            'manhattan_partial_both_k0k1': manhattan_partial_both,
            'obj': current_obj,
            'obj_gap_percent': obj_gap_percent,
            'bks': bks,
            'bks_time': bks_time,
            'violation_sum': violation_sum,
            'violation_avg': violation_avg,
            'num_constraints': num_constraints,
            'num_variables': num_variables,
            'violations': violations,
        }
        # 统计 SC 完整解的 0/1 比例以及真实解的 0/1 比例（基于 bks_solution）
        try:
            sc_pred_bins = [1 if float(v) >= 0.5 else 0 for v in vaule_dict.values()]
            sc_one_count = sum(sc_pred_bins)
            sc_zero_count = len(sc_pred_bins) - sc_one_count
            total_sc_vars = len(sc_pred_bins) if len(sc_pred_bins) > 0 else None
            sc_one_pct = (sc_one_count / total_sc_vars) if total_sc_vars else None
            sc_zero_pct = (sc_zero_count / total_sc_vars) if total_sc_vars else None
        except Exception:
            sc_one_count = None
            sc_zero_count = None
            sc_one_pct = None
            sc_zero_pct = None

        # true solution counts from bks_solution (keys like 'x0')
        try:
            true_vals = list(bks_solution.values()) if isinstance(bks_solution, dict) else []
            true_bins = [1 if float(v) >= 0.5 else 0 for v in true_vals]
            true_one_count_sc = sum(true_bins) if true_bins else None
            true_zero_count_sc = (len(true_bins) - true_one_count_sc) if true_bins else None
            total_true_sc = len(true_bins) if true_bins else None
            true_one_pct_sc = (true_one_count_sc / total_true_sc) if total_true_sc else None
            true_zero_pct_sc = (true_zero_count_sc / total_true_sc) if total_true_sc else None
        except Exception:
            true_one_count_sc = None
            true_zero_count_sc = None
            true_one_pct_sc = None
            true_zero_pct_sc = None

        # append counts to SC result
        results[lp_file].update({
            'sc_one_count': sc_one_count,
            'sc_zero_count': sc_zero_count,
            'sc_one_pct': sc_one_pct,
            'sc_zero_pct': sc_zero_pct,
            'true_one_count': true_one_count_sc,
            'true_zero_count': true_zero_count_sc,
            'true_one_pct': true_one_pct_sc,
            'true_zero_pct': true_zero_pct_sc,
        })

        # 如果提供了 ps_var_lists，计算 SC 在这些部分解上的 0/1 统计（按总变量数归一化）
        try:
            # sc_pred_dict: varname -> 0/1
            sc_pred_dict = {}
            num_vars = len(vaule_dict) if vaule_dict else None
            for k, v in vaule_dict.items():
                try:
                    sc_pred_dict[k] = 1 if float(v) >= 0.5 else 0
                except Exception:
                    sc_pred_dict[k] = 0

            if ps_var_lists is not None:
                lists = ps_var_lists.get(lp_file)
                if lists is None:
                    lists = ps_var_lists.get(os.path.basename(lp_file))
                if lists is not None:
                    top_k1_list = lists.get('top_k1', [])
                    bottom_k0_list = lists.get('bottom_k0', [])
                    sc_partial_top_one_count = sum(1 for n in top_k1_list if sc_pred_dict.get(n, 0) == 1)
                    sc_partial_top_one_pct = (sc_partial_top_one_count / float(num_vars)) if num_vars else None
                    sc_partial_bottom_zero_count = sum(1 for n in bottom_k0_list if sc_pred_dict.get(n, 1) == 0)
                    sc_partial_bottom_zero_pct = (sc_partial_bottom_zero_count / float(num_vars)) if num_vars else None
                    sc_partial_both_one_count = sum(1 for n in top_k1_list if (n not in bottom_k0_list) and sc_pred_dict.get(n, 0) == 1) # 莫名其妙，这个似乎不需要
                    sc_partial_both_one_pct = (sc_partial_both_one_count / float(num_vars)) if num_vars else None

                    # true partial counts based on bks_solution
                    true_partial_top_one_count_sc = sum(1 for n in top_k1_list if float(bks_solution.get(n, 0)) >= 0.5) if bks_solution else None
                    true_partial_top_one_pct_sc = (true_partial_top_one_count_sc / float(num_vars)) if (true_partial_top_one_count_sc is not None and num_vars) else None
                    true_partial_bottom_one_count_sc = sum(1 for n in bottom_k0_list if float(bks_solution.get(n, 0)) >= 0.5) if bks_solution else None
                    true_partial_bottom_one_pct_sc = (true_partial_bottom_one_count_sc / float(num_vars)) if (true_partial_bottom_one_count_sc is not None and num_vars) else None
                    true_partial_both_one_count_sc = sum(1 for n in set(top_k1_list + bottom_k0_list) if float(bks_solution.get(n, 0)) >= 0.5) if bks_solution else None
                    true_partial_both_one_pct_sc = (true_partial_both_one_count_sc / float(num_vars)) if (true_partial_both_one_count_sc is not None and num_vars) else None

                    results[lp_file].update({
                        'sc_partial_top_one_count': sc_partial_top_one_count,
                        'sc_partial_top_one_pct': sc_partial_top_one_pct,
                        'sc_partial_bottom_zero_count': sc_partial_bottom_zero_count,
                        'sc_partial_bottom_zero_pct': sc_partial_bottom_zero_pct,
                        'sc_partial_both_one_count': sc_partial_both_one_count,
                        'sc_partial_both_one_pct': sc_partial_both_one_pct,
                        'true_partial_top_one_count': true_partial_top_one_count_sc,
                        'true_partial_top_one_pct': true_partial_top_one_pct_sc,
                        'true_partial_bottom_one_count': true_partial_bottom_one_count_sc,
                        'true_partial_bottom_one_pct': true_partial_bottom_one_pct_sc,
                        'true_partial_both_one_count': true_partial_both_one_count_sc,
                        'true_partial_both_one_pct': true_partial_both_one_pct_sc,
                    })
        except Exception:
            pass

    out_path = os.path.join(config_folder, f'manhattan_results_{task_name}_sc.json')
    with open(out_path, 'w', encoding='utf-8') as of:
        json.dump(results, of, indent=2)
    return results




# 先运行 PS，获取部分解变量列表，然后将其传入 SC 做对比
ps_res, ps_var_lists = compute_ps_manhattan_distance(DEFAULT_TASK_NAME, DEFAULT_TEST_NUM, DEFAULT_K0, DEFAULT_K1, DEFAULT_DELTA)
sc_res = compute_sc_manhattan_distance(
    task_name=DEFAULT_TASK_NAME,
    agg_num=50,
    k0=DEFAULT_K0,
    k1=DEFAULT_K1,
    Delta=DEFAULT_DELTA,
    ps_var_lists=ps_var_lists,
)

# 汇总并保存为 CSV：比较 PS 与 SC 的曼哈顿距离及归一化曼哈顿距离（使用字典写法保证列对齐）
try:
    ps_results = ps_res
    sc_results = sc_res
    csv_path = os.path.join(config_folder, f'manhattan_summary_{DEFAULT_TASK_NAME}_{timestamp}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        fieldnames = [
            "instance",
            "ps_manhattan",
            "sc_manhattan",
            "ps_norm_manhattan",
            "sc_norm_manhattan",
            "ps_partial_top",
            "ps_partial_bottom",
            "ps_partial_both",
            "ps_partial_top_norm",
            "ps_partial_bottom_norm",
            "ps_partial_both_norm",
            "sc_partial_top",
            "sc_partial_bottom",
            "sc_partial_both",
            "sc_partial_top_norm",
            "sc_partial_bottom_norm",
            "sc_partial_both_norm",
            "ps_obj",
            "ps_obj_gap",
            "sc_obj",
            "sc_obj_gap",
            "ps_violation_sum",
            "sc_violation_sum",
            "ps_violation_norm_by_ps_constraints",
            "sc_violation_norm_by_ps_constraints",
            "ps_one_count",
            "ps_zero_count",
            "ps_one_pct",
            "ps_zero_pct",
            "sc_one_count",
            "sc_zero_count",
            "sc_one_pct",
            "sc_zero_pct",
            # PS 部分解计数（按总变量数归一化）
            "ps_partial_top_one_count",
            "ps_partial_top_one_pct",
            "ps_partial_bottom_zero_count",
            "ps_partial_bottom_zero_pct",
            "ps_partial_both_one_count",
            "ps_partial_both_one_pct",
            # SC 部分解计数（按总变量数归一化）
            "sc_partial_top_one_count",
            "sc_partial_top_one_pct",
            "sc_partial_bottom_zero_count",
            "sc_partial_bottom_zero_pct",
            "sc_partial_both_one_count",
            "sc_partial_both_one_pct",
            # 最优解的 0/1 统计：方法无关，统一为单列
            "true_one_count",
            "true_zero_count",
            "true_one_pct",
            "true_zero_pct",
            # 真实解在部分解集合上的统计（按总变量数归一化）
            "true_partial_top_one_count",
            "true_partial_top_one_pct",
            "true_partial_bottom_one_count",
            "true_partial_bottom_one_pct",
            "true_partial_both_one_count",
            "true_partial_both_one_pct",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        all_instances = sorted(set(list(ps_results.keys()) + list(sc_results.keys())))
        def fmt(x):
            return "" if x is None else (f"{x:.6f}" if isinstance(x, float) else str(x))

        for inst in all_instances:
            ps_entry = ps_results.get(inst, {}) if isinstance(ps_results.get(inst), dict) else {}
            sc_entry = sc_results.get(inst, {}) if isinstance(sc_results.get(inst), dict) else {}

            ps_m = ps_entry.get('manhattan') if isinstance(ps_entry, dict) else None
            sc_m = sc_entry.get('manhattan') if isinstance(sc_entry, dict) else None

            # 变量数优先使用 SC 的值
            num_vars = None
            if isinstance(sc_entry, dict) and sc_entry.get('num_variables'):
                num_vars = sc_entry.get('num_variables')
            elif isinstance(ps_entry, dict) and ps_entry.get('num_variables'):
                num_vars = ps_entry.get('num_variables')

            ps_partial_top = ps_entry.get('manhattan_partial_top_k1') if isinstance(ps_entry, dict) else None
            ps_partial_bottom = ps_entry.get('manhattan_partial_bottom_k0') if isinstance(ps_entry, dict) else None
            ps_partial_both = ps_entry.get('manhattan_partial_both_k0k1') if isinstance(ps_entry, dict) else None
            sc_partial_top = sc_entry.get('manhattan_partial_top_k1') if isinstance(sc_entry, dict) else None
            sc_partial_bottom = sc_entry.get('manhattan_partial_bottom_k0') if isinstance(sc_entry, dict) else None
            sc_partial_both = sc_entry.get('manhattan_partial_both_k0k1') if isinstance(sc_entry, dict) else None

            ps_partial_top_norm = (float(ps_partial_top) / float(num_vars)) if (ps_partial_top is not None and num_vars) else None
            ps_partial_bottom_norm = (float(ps_partial_bottom) / float(num_vars)) if (ps_partial_bottom is not None and num_vars) else None
            ps_partial_both_norm = (float(ps_partial_both) / float(num_vars)) if (ps_partial_both is not None and num_vars) else None
            sc_partial_top_norm = (float(sc_partial_top) / float(num_vars)) if (sc_partial_top is not None and num_vars) else None
            sc_partial_bottom_norm = (float(sc_partial_bottom) / float(num_vars)) if (sc_partial_bottom is not None and num_vars) else None
            sc_partial_both_norm = (float(sc_partial_both) / float(num_vars)) if (sc_partial_both is not None and num_vars) else None

            ps_norm = (float(ps_m) / float(num_vars)) if (ps_m is not None and num_vars) else None
            sc_norm = (float(sc_m) / float(num_vars)) if (sc_m is not None and num_vars) else None

            ps_obj = ps_entry.get('obj') if isinstance(ps_entry, dict) else None
            ps_obj_gap = ps_entry.get('obj_gap_percent') if isinstance(ps_entry, dict) else None
            sc_obj = sc_entry.get('obj') if isinstance(sc_entry, dict) else None
            sc_obj_gap = sc_entry.get('obj_gap_percent') if isinstance(sc_entry, dict) else None

            ps_violation_sum = ps_entry.get('violation_sum') if isinstance(ps_entry, dict) else None
            sc_violation_sum = sc_entry.get('violation_sum') if isinstance(sc_entry, dict) else None

            # 违反度归一化：除以 PS 的约束数量（如果可用）
            ps_num_constraints = ps_entry.get('num_constraints') if isinstance(ps_entry, dict) else None
            try:
                ps_violation_norm = (float(ps_violation_sum) / float(ps_num_constraints)) if (ps_violation_sum is not None and ps_num_constraints) else None
            except Exception:
                ps_violation_norm = None
            try:
                sc_violation_norm = (float(sc_violation_sum) / float(ps_num_constraints)) if (sc_violation_sum is not None and ps_num_constraints) else None
            except Exception:
                sc_violation_norm = None

            row = {
                "instance": inst,
                "ps_manhattan": fmt(ps_m),
                "sc_manhattan": fmt(sc_m),
                "ps_norm_manhattan": fmt(ps_norm),
                "sc_norm_manhattan": fmt(sc_norm),
                "ps_partial_top": fmt(ps_partial_top),
                "ps_partial_bottom": fmt(ps_partial_bottom),
                "ps_partial_both": fmt(ps_partial_both),
                "ps_partial_top_norm": fmt(ps_partial_top_norm),
                "ps_partial_bottom_norm": fmt(ps_partial_bottom_norm),
                "ps_partial_both_norm": fmt(ps_partial_both_norm),
                "sc_partial_top": fmt(sc_partial_top),
                "sc_partial_bottom": fmt(sc_partial_bottom),
                "sc_partial_both": fmt(sc_partial_both),
                "sc_partial_top_norm": fmt(sc_partial_top_norm),
                "sc_partial_bottom_norm": fmt(sc_partial_bottom_norm),
                "sc_partial_both_norm": fmt(sc_partial_both_norm),
                "ps_obj": fmt(ps_obj),
                "ps_obj_gap": fmt(ps_obj_gap),
                "sc_obj": fmt(sc_obj),
                "sc_obj_gap": fmt(sc_obj_gap),
                "ps_violation_sum": fmt(ps_violation_sum),
                "sc_violation_sum": fmt(sc_violation_sum),
                "ps_violation_norm_by_ps_constraints": fmt(ps_violation_norm),
                "sc_violation_norm_by_ps_constraints": fmt(sc_violation_norm),
                "ps_one_count": fmt(ps_entry.get('ps_one_count') if isinstance(ps_entry, dict) else None),
                "ps_zero_count": fmt(ps_entry.get('ps_zero_count') if isinstance(ps_entry, dict) else None),
                "ps_one_pct": fmt(ps_entry.get('ps_one_pct') if isinstance(ps_entry, dict) else None),
                "ps_zero_pct": fmt(ps_entry.get('ps_zero_pct') if isinstance(ps_entry, dict) else None),
                "sc_one_count": fmt(sc_entry.get('sc_one_count') if isinstance(sc_entry, dict) else None),
                "sc_zero_count": fmt(sc_entry.get('sc_zero_count') if isinstance(sc_entry, dict) else None),
                "sc_one_pct": fmt(sc_entry.get('sc_one_pct') if isinstance(sc_entry, dict) else None),
                "sc_zero_pct": fmt(sc_entry.get('sc_zero_pct') if isinstance(sc_entry, dict) else None),
                "ps_partial_top_one_count": fmt(ps_entry.get('ps_partial_top_one_count') if isinstance(ps_entry, dict) else None),
                "ps_partial_top_one_pct": fmt(ps_entry.get('ps_partial_top_one_pct') if isinstance(ps_entry, dict) else None),
                "ps_partial_bottom_zero_count": fmt(ps_entry.get('ps_partial_bottom_zero_count') if isinstance(ps_entry, dict) else None),
                "ps_partial_bottom_zero_pct": fmt(ps_entry.get('ps_partial_bottom_zero_pct') if isinstance(ps_entry, dict) else None),
                "ps_partial_both_one_count": fmt(ps_entry.get('ps_partial_both_one_count') if isinstance(ps_entry, dict) else None),
                "ps_partial_both_one_pct": fmt(ps_entry.get('ps_partial_both_one_pct') if isinstance(ps_entry, dict) else None),
                "sc_partial_top_one_count": fmt(sc_entry.get('sc_partial_top_one_count') if isinstance(sc_entry, dict) else None),
                "sc_partial_top_one_pct": fmt(sc_entry.get('sc_partial_top_one_pct') if isinstance(sc_entry, dict) else None),
                "sc_partial_bottom_zero_count": fmt(sc_entry.get('sc_partial_bottom_zero_count') if isinstance(sc_entry, dict) else None),
                "sc_partial_bottom_zero_pct": fmt(sc_entry.get('sc_partial_bottom_zero_pct') if isinstance(sc_entry, dict) else None),
                "sc_partial_both_one_count": fmt(sc_entry.get('sc_partial_both_one_count') if isinstance(sc_entry, dict) else None),
                "sc_partial_both_one_pct": fmt(sc_entry.get('sc_partial_both_one_pct') if isinstance(sc_entry, dict) else None),
            }

            # 统一最优解统计（优先使用 PS 的统计，否则使用 SC 的）
            true_one = None
            true_zero = None
            true_one_pct = None
            true_zero_pct = None
            if isinstance(ps_entry, dict) and ps_entry.get('true_one_count') is not None:
                true_one = ps_entry.get('true_one_count')
                true_zero = ps_entry.get('true_zero_count')
                true_one_pct = ps_entry.get('true_one_pct')
                true_zero_pct = ps_entry.get('true_zero_pct')
            elif isinstance(sc_entry, dict) and sc_entry.get('true_one_count') is not None:
                true_one = sc_entry.get('true_one_count')
                true_zero = sc_entry.get('true_zero_count')
                true_one_pct = sc_entry.get('true_one_pct')
                true_zero_pct = sc_entry.get('true_zero_pct')

            row.update({
                "true_one_count": fmt(true_one),
                "true_zero_count": fmt(true_zero),
                "true_one_pct": fmt(true_one_pct),
                "true_zero_pct": fmt(true_zero_pct),
            })

            # 真实解在部分解集合上的统计：优先使用 PS 提供的部分统计，否则使用 SC 的
            true_partial_top_one = None
            true_partial_bottom_one = None
            true_partial_both_one = None
            true_partial_top_one_pct = None
            true_partial_bottom_one_pct = None
            true_partial_both_one_pct = None
            if isinstance(ps_entry, dict) and ps_entry.get('true_partial_top_one_count') is not None:
                true_partial_top_one = ps_entry.get('true_partial_top_one_count')
                true_partial_bottom_one = ps_entry.get('true_partial_bottom_one_count')
                true_partial_both_one = ps_entry.get('true_partial_both_one_count')
                true_partial_top_one_pct = ps_entry.get('true_partial_top_one_pct')
                true_partial_bottom_one_pct = ps_entry.get('true_partial_bottom_one_pct')
                true_partial_both_one_pct = ps_entry.get('true_partial_both_one_pct')
            elif isinstance(sc_entry, dict) and sc_entry.get('true_partial_top_one_count') is not None:
                true_partial_top_one = sc_entry.get('true_partial_top_one_count')
                true_partial_bottom_one = sc_entry.get('true_partial_bottom_one_count')
                true_partial_both_one = sc_entry.get('true_partial_both_one_count')
                true_partial_top_one_pct = sc_entry.get('true_partial_top_one_pct')
                true_partial_bottom_one_pct = sc_entry.get('true_partial_bottom_one_pct')
                true_partial_both_one_pct = sc_entry.get('true_partial_both_one_pct')

            row.update({
                "true_partial_top_one_count": fmt(true_partial_top_one),
                "true_partial_top_one_pct": fmt(true_partial_top_one_pct),
                "true_partial_bottom_one_count": fmt(true_partial_bottom_one),
                "true_partial_bottom_one_pct": fmt(true_partial_bottom_one_pct),
                "true_partial_both_one_count": fmt(true_partial_both_one),
                "true_partial_both_one_pct": fmt(true_partial_both_one_pct),
            })

            writer.writerow(row)
    print(f"Wrote summary CSV: {csv_path}")
except Exception as e:
    print("Failed to write summary CSV:", e)












