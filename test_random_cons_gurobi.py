import gurobipy as gp
from gurobipy import Model, GRB
import os
# from ecole import make_env
# from ecole.core import Observation
# from ecole.scip import Model
import numpy as np
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

def solve_lp_files_gurobi(directory: str, num_problems: int, agg_num: int, delete_con:bool, seed:int):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    primes = gen_primes(agg_num)
    u_list = [math.log(p) for p in primes]

    results = []

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        lp_path = os.path.join(directory, lp_file)
        print(f"Processing {lp_file}")

        ## 原问题
        model_orig = gp.read(lp_path)
        # 时间从读入开始算
        t0 = time.perf_counter()
        model_orig.optimize()
        t1 = time.perf_counter()
        status_orig = model_orig.Status
        obj_orig = model_orig.ObjVal
        time_orig = t1 - t0


        ## 聚合问题
        model_agg = gp.read(lp_path)
        t2 = time.perf_counter()
        conss = model_agg.getConstrs()
        cons_num_orig = model_agg.NumConstrs
        original_cons_num = len(conss)
        sample = random.sample(conss, min(agg_num, len(conss)))

        agg_coeffs = {}
        agg_rhs = 0.0

        for idx, cons in enumerate(sample):
            u = u_list[idx]
            constr_expr = model_agg.getRow(cons)
            for j in range(constr_expr.size()):
                var = constr_expr.getVar(j)
                coef = constr_expr.getCoeff(j)
                agg_coeffs[var.VarName] = agg_coeffs.get(var.VarName, 0.0) + u * coef
            agg_rhs += u * cons.RHS
            if delete_con:
                model_agg.remove(cons)

        model_agg.update()

        # 构造聚合约束
        expr = 0
        for var_name, coef in agg_coeffs.items():
            var = model_agg.getVarByName(var_name)
            expr += coef * var
        model_agg.addConstr(expr <= agg_rhs, name="agg_constraint")
        model_agg.update()

        cons_num_agg = model_agg.NumConstrs
        model_agg.optimize()
        t3 = time.perf_counter()
        status_agg = model_agg.Status
        obj_agg = model_agg.ObjVal
        time_agg = t3 - t2
        after_agg_cons_num = len(model_agg.getConstrs())

        dual_gap = (obj_agg - obj_orig) / abs(obj_orig) if obj_orig != 0 else float("inf")
        time_reduce = (time_orig - time_agg) / time_orig if time_orig > 0 else 0

        print(f"原obj:{obj_orig},\t 聚合后obj：{obj_agg}")
        print(f"原时间:{time_orig},\t 聚合后时间:{time_agg}")
        print(f"gap:{dual_gap}")
        print(f"time_reduce:{time_reduce}")

        results.append({
            "filename": lp_file,
            "status_orig": status_orig,
            "status_agg": status_agg,
            "obj_orig": obj_orig,
            "obj_agg": obj_agg,
            "dual_gap": dual_gap,
            "original_cons_num": cons_num_orig,
            "after_cons_num":cons_num_agg,
            "cons_reduce_ratio":(cons_num_orig-cons_num_agg)/cons_num_orig,
            "time_orig": time_orig,
            "time_agg": time_agg,
            "time_reduce": time_reduce,
            "original_cons": original_cons_num,
            "after_agg_cons": after_agg_cons_num,
            "seed":seed,
        })

    df = pd.DataFrame(results)
    return df

if __name__ == '__main__':
    data_dir = "CA_200_400"
    lp_files_dir = f"./instance/test/{data_dir}"
    solve_num = 50
    agg_num = 15
    delete_con = True
    result_dir = f"./result/{data_dir}_agg_num_{agg_num}_delete_con_{str(delete_con)}"
    # 第一个是20的约束
    # 第二个490，取100吧


    seed_list = [1,2,3,4,5,6,7,8,9,10]
    all_runs = []
    gurobi_solve = True
    os.makedirs(result_dir, exist_ok=True)
    for seed in seed_list:
        random.seed(seed)
        if gurobi_solve:
            df = solve_lp_files_gurobi(lp_files_dir, solve_num, agg_num,delete_con,seed)
            df.to_csv(result_dir + f"/surrogate_stats_aggnum_{agg_num}_seed_{seed}.csv", index=False)
            all_runs.append(df)

    # 合并所有 seed 的明细
    df_all = pd.concat(all_runs, ignore_index=True)
    df_all.to_csv(os.path.join(result_dir, "all_details.csv"), index=False)
    # 按 filename 汇总统计
    summary = df_all.groupby("filename").agg({
        "dual_gap":    ["min","max","mean"],
        "time_reduce": ["min","max","mean"],
        "cons_reduce_ratio": ["min","max","mean"]
    })
    # 扁平化多级列名
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.reset_index(inplace=True)
    summary.to_csv(os.path.join(result_dir, "surrogate_summary.csv"), index=False)

    print("Done. 明细和摘要已保存至：", result_dir)


