import gurobipy as gp
from gurobipy import Model, GRB, LinExpr
import os
# from ecole import make_env
# from ecole.core import Observation
# from ecole.scip import Model
import numpy as np
import math
import time
import random
import pandas as pd
import json
import repair_and_post_solve_func
import utils

k0 = 30
k1 = 30
Delta = 10


def solve_lp_files_gurobi(cache:dict,directory: str, num_problems: int, agg_num: int,seed:int,problem:str,repair_method:str,PostSolve:bool,agg_model_solve_time,time_limit):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # 生成乘子
    # todo:在此处设置乘子？
    primes = utils.gen_primes(agg_num)
    # u_list = [math.log(p) for p in primes]
    u_list = [1 for p in primes]

    results = []

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        # 得到lp路径
        lp_path = os.path.join(directory, lp_file)
        print(f"Processing {lp_file}")

        ## 原问题求解
        # 如果缓存中已有结果，就直接读取，否则求解并写入缓存
        if lp_path in cache:
            print("------------read cache-------------")
            entry = cache[lp_path]
            obj_sense = entry['obj_sense']
            status_orig = entry['status_orig']
            obj_orig = entry['obj_orig']
            time_orig = entry['time_orig']
        else:
            raise Exception("there is not cache")

        ## 聚合问题求解
        # 读入问题
        model_agg = gp.read(lp_path)
        t0 = time.perf_counter()

        # 指标
        cons_num_orig = model_agg.NumConstrs

        # 聚合
        utils.aggregate_constr(model_agg,agg_num)

        # 聚合后约束数量
        cons_num_agg = model_agg.NumConstrs

        ## 一些求解的参数 todo
        # 解数量，时间限制
        # model_agg.setParam('SolutionLimit', 1)
        print("------------solving agg model-------------")
        model_agg.setParam("Threads",Threads)
        model_agg.setParam("Seed", seed)
        if agg_model_solve_time == -1:
            model_agg.optimize()
        else:
            model_agg.setParam("TimeLimit", agg_model_solve_time)
            model_agg.optimize()
        agg_objval_original = model_agg.ObjVal

        # 获得变量值
        Vars = model_agg.getVars()
        vaule_dict = {var.VarName: var.X for var in Vars}

        # 读入新模型，用于解的可行性修复
        repair_model = gp.read(lp_path)
        repair_model.setParam("Threads", Threads)
        repair_model.setParam("Seed", seed+1)
        if repair_method == "naive":
            # 简单的启发式修复
            vaule_dict = repair_and_post_solve_func.heuristic_repair(repair_model, vaule_dict)
        elif repair_method == "score":
            # 变量评分
            vaule_dict = repair_and_post_solve_func.heuristic_repair_with_score(repair_model, vaule_dict)
        elif repair_method == "subproblem":
            vaule_dict = repair_and_post_solve_func.heuristic_repair_subproblem(repair_model, vaule_dict)
        elif repair_method == "lightmilp":
            vaule_dict = repair_and_post_solve_func.heuristic_repair_light_MILP(repair_model, vaule_dict, lp_path)
        else:
            raise Exception("unknown repair_method")


        # 得到可行解后，后处理
        repair_and_post_solve_func.PostSolve(repair_model,k0,k1,Delta,vaule_dict,lp_file,t0,time_limit)
        t1 = time.perf_counter()

        status_agg = repair_model.Status
        obj_agg = repair_model.ObjVal
        total_time_agg = t1 - t0

        ## gap、时间约简计算
        if obj_sense == GRB.MINIMIZE:
            print("最小化问题")
            primal_gap = (obj_agg - obj_orig) / abs(obj_orig) if obj_orig != 0 else float("inf")
        else:
            print("最大化问题")
            primal_gap = (obj_orig - obj_agg) / abs(obj_orig) if obj_orig != 0 else float("inf")
        time_reduce = (time_orig - total_time_agg) / time_orig if time_orig > 0 else 0

        print(f"原obj:{obj_orig},\t 聚合后obj：{obj_agg}")
        print(f"原时间:{time_orig},\t 聚合后时间:{total_time_agg}")
        print(f"primal_gap:{primal_gap}")
        print(f"time_reduce:{time_reduce}")
        # 保存
        results.append({
            "filename": lp_file,
            "status_orig": status_orig,
            "status_agg": status_agg,
            "obj_orig": obj_orig,
            "obj_agg": obj_agg,
            "primal_gap": primal_gap,
            "original_cons_num": cons_num_orig,
            "after_cons_num": cons_num_agg,
            "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
            "time_orig": time_orig,
            "time_agg": total_time_agg,
            "time_reduce": time_reduce,
            "original_cons": cons_num_orig,
            "after_agg_cons": cons_num_agg,
        })
        # 和原始时间对比

    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':

    problem = "CA"
    json_file_path = 'parameters/param_2.json'
    params = utils.get_problem_parameters(json_file_path)
    post_fix = utils.get_post_fix(params[problem])

    data_dir = "_".join([problem,post_fix])
    # data_dir = "CA_500_600"

    # lp_files_dir = f"./instance/test/{data_dir}"
    dataset_name = "test"
    lp_files_dir = f"./instance/{dataset_name}/{data_dir}"
    seed_list = [2,3,4,5,6]
    solve_num = 10
    agg_num = 10
    Threads = 0 # default 0
    # Threads = 5
    # agg_model_solve_time = -1 # 不限制时间
    agg_model_solve_time = 2
    time_limit = 1000
    repair_method_list = ["None", "gurobi", "naive", "score", "subproblem","lightmilp"]
    post_solve = True
    for idx in range(4,5):

        repair_method = repair_method_list[idx]

        result_dir = f"./result/random_{data_dir}_solve_{solve_num}_aggNum_{agg_num}_aggsolvetime_{agg_model_solve_time}_repair_{repair_method}_PostSolve_{post_solve}_Threads_{Threads}"

        # 保存统计结果
        all_runs = []
        os.makedirs(result_dir, exist_ok=True)

        # 读取cache
        # todo：求最优解时是可以并行求解的，但是聚合求解不是。
        cache_dir = f"./cache/{dataset_name}"
        cache = utils.load_optimal_cache(cache_dir, data_dir, lp_files_dir, solve_num, Threads,time_limit)

        # 每个seed，求解一次
        for seed in seed_list:
            random.seed(seed)
            df = solve_lp_files_gurobi(cache, lp_files_dir, solve_num, agg_num, seed, problem, repair_method, post_solve, agg_model_solve_time,time_limit)
            df.to_csv(result_dir + f"/random_aggnum_{agg_num}_seed_{seed}_repair_{repair_method}.csv", index=False)
            all_runs.append(df)

        # 合并所有 seed 的明细
        df_all = pd.concat(all_runs, ignore_index=True)
        df_all.to_csv(os.path.join(result_dir, "all_details.csv"), index=False)

        # 按 filename 汇总统计
        if repair_method != "None":
            # 修复，primal gap
            summary = df_all.groupby("filename").agg({
                "primal_gap": ["min", "max", "mean"],
                "time_reduce": ["min", "max", "mean"],
                "cons_reduce_ratio": ["min", "max", "mean"]
            })
        else:
            # 不修复，dual gap
            summary = df_all.groupby("filename").agg({
                "dual_gap":    ["min","max","mean"],
                "time_reduce": ["min","max","mean"],
                "cons_reduce_ratio": ["min","max","mean"]
            })
        # 扁平化多级列名
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary.reset_index(inplace=True)
        summary.to_csv(os.path.join(result_dir, "surrogate_summary.csv"), index=False)

        print("明细和摘要已保存至：", result_dir)


