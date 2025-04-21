import gurobipy as gp
from gurobipy import Model, GRB
import os
# from ecole import make_env
# from ecole.core import Observation
# from ecole.scip import Model
import numpy as np
from pyscipopt import Model, SCIP_STATUS, quicksum
import pyscipopt
import math
import time
import random
import pandas as pd

def gen_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % p != 0 for p in primes):
            primes.append(num)
        num += 1
    return primes

def solve_lp_files_gurobi(directory: str, num_problems: int):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        lp_path = os.path.join(directory, lp_file)

        try:
            # 创建 Gurobi 模型
            model = gp.read(lp_path)

            # 求解模型
            model.optimize()

            # 获取求解状态
            status = model.Status

            # 获取目标值
            objective_value = model.ObjVal if status == GRB.OPTIMAL else None

            # 输出结果
            print(f"文件: {lp_file}")
            print(f"求解状态: {status}")
            print(f"目标值: {objective_value}")
            print("-" * 40)

        except gp.GurobiError as e:
            print(f"求解文件 {lp_file} 时出错: {e}")
            print("-" * 40)

def solve_lp_files_scip(directory: str, num_problems: int, agg_cons_num: int):
    # 列出所有 .lp 文件并排序
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()
    lp_files = lp_files[:num_problems]

    primes = gen_primes(agg_cons_num)
    u_list = [math.log(p) for p in primes]
    # u_list = [p for p in primes]
    results = []

    for lp_file in lp_files:
        lp_path = os.path.join(directory, lp_file)
        print(f"Processing {lp_file}")

        # ===== 原问题求解 =====
        model_orig = Model()
        model_orig.readProblem(lp_path)
        # 从读进来开始算时间起点
        t0 = time.perf_counter()
        model_orig.optimize()
        t1 = time.perf_counter()
        status_orig = model_orig.getStatus()
        obj_orig = model_orig.getObjVal()
        time_orig = t1 - t0

        # ===== 聚合约束后的问题 =====
        model_agg = Model()
        model_agg.readProblem(lp_path)
        # 从读进来开始算时间起点
        t2 = time.perf_counter()
        conss = model_agg.getConss()
        original_cons_num = len(conss)
        sample = random.sample(conss, min(agg_cons_num, len(conss)))

        # 名字→对象的映射
        var_map = {var.name: var for var in model_agg.getVars()}

        # 计算聚合系数和 RHS
        agg_coeffs = {}
        agg_rhs = 0.0
        for idx, cons in enumerate(sample):
            u = u_list[idx]
            row = model_agg.getValsLinear(cons)
            rhs = model_agg.getRhs(cons)
            for var, coef in row.items():
                agg_coeffs[var] = agg_coeffs.get(var, 0.0) + u * coef
            agg_rhs += u * rhs
            model_agg.delCons(cons)  # 删除原始约束

        # 构造线性表达式
        expr = quicksum(coef *  var_map[var_name] for var_name, coef in agg_coeffs.items())
        model_agg.addCons(expr <= agg_rhs)

        # # 求解聚合问题
        conss = model_agg.getConss()
        after_agg_cons_num = len(conss)
        model_agg.optimize()
        t3 = time.perf_counter()
        status_agg = model_agg.getStatus()
        obj_agg = model_agg.getObjVal()
        time_agg = t3 - t2

        # # 输出对比结果
        print(f"文件: {lp_file}")
        print(f" 原问题: 状态={status_orig}, 目标={obj_orig:.4f}, 用时={time_orig:.4f}s")
        print(f" 聚合问题: 状态={status_agg}, 目标={obj_agg:.4f}, 用时={time_agg:.4f}s")
        print(f"原约束个数：{original_cons_num}, 约简后约束个数：{after_agg_cons_num}")
        dual_gap = (obj_agg - obj_orig)/obj_orig
        print(f"dual gap: {dual_gap}")
        print("-" * 50)

        # 存入结果表
        results.append({
            "filename": lp_file,
            "status_orig": status_orig,
            "status_agg": status_agg,
            "obj_orig": obj_orig,
            "obj_agg": obj_agg,
            "dual_gap": dual_gap,
            "time_orig": time_orig,
            "time_agg": time_agg,
            "time_reduce": (time_orig-time_agg)/time_orig,
            "original_cons": original_cons_num,
            "after_agg_cons": after_agg_cons_num,
        })

    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    lp_files_dir = "./instance/test/CA"
    solve_num = 50
    agg_num = 20
    seed = random.randint(0,99999)
    random.seed(seed)
    # solve_lp_files_gurobi(lp_files_dir,solve_num)
    df = solve_lp_files_scip(lp_files_dir,solve_num,agg_num)
    df.to_csv(f"surrogate_stats_aggnum_{agg_num}_seed_{seed}.csv", index=False)


