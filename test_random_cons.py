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
import concurrent.futures

# k0 = 100
# k1 = 100
# Delta = 10


def accelerated_solving(cache:dict, directory: str, num_problems: int, agg_num: int, seed:int, problem:str, repair_method:str, agg_model_solve_time, time_limit,neighborhood,Threads):
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
        # 如果缓存中已有结果，就直接读取，否则报错
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
        repair_and_post_solve_func.PostSolve(repair_model,neighborhood,vaule_dict,lp_file,t0,time_limit)
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


def solve_single_problem_pipeline(
        lp_path: str,
        lp_file: str,
        cache: dict,
        agg_num: int,
        seed: int,
        repair_method: str,
        agg_model_solve_time: float,
        time_limit: float,
        threads_per_solve: int,  # Gurobi 内部线程数
        k0,
        k1,
        Delta
):
    """
    单个进程的工作单元：对单个 LP 文件执行聚合、求解和修复流程。
    返回单个问题的结果字典，或返回 None 表示失败。
    """
    try:
        print(f"[PID:{os.getpid()}] Processing {lp_file} with seed {seed}")

        # ----------------- 1. 读取原问题缓存 -----------------
        if lp_path not in cache:
            print(f"[PID:{os.getpid()}] 错误: {lp_file} 缺少缓存，跳过。")
            return None

        entry = cache[lp_path]
        obj_sense = entry['obj_sense']
        status_orig = entry['status_orig']
        obj_orig = entry['obj_orig']
        time_orig = entry['time_orig']

        # ----------------- 2. 聚合问题求解 -----------------

        # 读入问题，开始计时
        model_agg = gp.read(lp_path)
        t0 = time.perf_counter()

        # 指标
        cons_num_orig = model_agg.NumConstrs

        # 聚合
        # 注意: 乘子 u_list 在这里不再需要，因为它们在 aggregate_constr 中使用
        utils.aggregate_constr(model_agg, agg_num)
        cons_num_agg = model_agg.NumConstrs

        # 求解参数设置
        model_agg.setParam("Threads", threads_per_solve)
        model_agg.setParam("Seed", seed)
        if agg_model_solve_time != -1:
            model_agg.setParam("TimeLimit", agg_model_solve_time)

        print(f"[PID:{os.getpid()}] Solving aggregated model...")
        model_agg.optimize()
        agg_objval_original = model_agg.ObjVal if model_agg.Status != GRB.INF_OR_UNBD else None

        # 获得变量值
        Vars = model_agg.getVars()
        vaule_dict = {var.VarName: var.X for var in Vars}

        # ----------------- 3. 可行性修复 -----------------

        repair_model = gp.read(lp_path)
        repair_model.setParam("Threads", threads_per_solve)
        repair_model.setParam("Seed", seed + 1)

        if repair_method == "naive":
            vaule_dict = repair_and_post_solve_func.heuristic_repair(repair_model, vaule_dict)
        elif repair_method == "score":
            vaule_dict = repair_and_post_solve_func.heuristic_repair_with_score(repair_model, vaule_dict)
        elif repair_method == "subproblem":
            vaule_dict = repair_and_post_solve_func.heuristic_repair_subproblem(repair_model, vaule_dict)
        elif repair_method == "lightmilp":
            vaule_dict = repair_and_post_solve_func.heuristic_repair_light_MILP(repair_model, vaule_dict, lp_path)
        else:
            raise Exception("unknown repair_method")

        # ----------------- 4. 后处理和结果记录 -----------------

        # 后处理
        repair_and_post_solve_func.PostSolve(repair_model, k0, k1, Delta, vaule_dict, lp_file, t0, time_limit)
        t1 = time.perf_counter()

        status_agg = repair_model.Status
        obj_agg = repair_model.ObjVal if status_agg != GRB.INF_OR_UNBD else None
        total_time_agg = t1 - t0

        # gap、时间约简计算
        if obj_agg is not None and obj_orig is not None and abs(obj_orig) > 1e-6:
            if obj_sense == GRB.MINIMIZE:
                primal_gap = (obj_agg - obj_orig) / abs(obj_orig)
            else:
                primal_gap = (obj_orig - obj_agg) / abs(obj_orig)
        else:
            primal_gap = float("inf")

        time_reduce = (time_orig - total_time_agg) / time_orig if time_orig > 0 else 0

        # 封装结果
        return {
            "filename": lp_file,
            "seed": seed,  # 新增 seed，方便追踪
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
        }

    except Exception as e:
        print(f"[PID:{os.getpid()}] 求解 {lp_file} 失败: {e}")
        return None






def solve_lp_files_gurobi_parallel(
        cache: dict,
        directory: str,
        num_problems: int,
        agg_num: int,
        seed: int,
        problem: str,
        repair_method: str,
        PostSolve: bool,
        agg_model_solve_time: float,
        time_limit: float,
        threads_per_solve: int,  # 每个实例的线程数
        num_parallel_solves: int,  # 并行实例数
        k0,
        k1,
        Delta
) -> pd.DataFrame:
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()
    lp_files = lp_files[:num_problems]

    print(f"--- Seed {seed}: 准备求解 {len(lp_files)} 个问题，并行 {num_parallel_solves} 个进程 ---")

    results = []

    # 使用 ProcessPoolExecutor 进行并行求解
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_solves) as executor:

        future_to_lp_file = {}

        # 提交所有任务
        for lp_file in lp_files:
            lp_path = os.path.join(directory, lp_file)

            # 为每个问题提交一个任务
            future = executor.submit(
                solve_single_problem_pipeline,
                lp_path,
                lp_file,
                cache,
                agg_num,
                seed,
                repair_method,
                agg_model_solve_time,
                time_limit,
                threads_per_solve,
                k0, k1, Delta
            )
            future_to_lp_file[future] = lp_file

        # 收集结果
        for future in concurrent.futures.as_completed(future_to_lp_file):
            lp_file = future_to_lp_file[future]
            try:
                result_dict = future.result()
                if result_dict:
                    results.append(result_dict)
                    print(f"✅ 完成并收集结果: {lp_file}")
                else:
                    print(f"❌ {lp_file} 求解失败或跳过。")
            except Exception as exc:
                print(f'🔴 {lp_file} 进程产生异常: {exc}')

    # 4. 转换为 DataFrame 并返回
    df = pd.DataFrame(results)
    return df




# 假设这个函数包含您的顶层实验逻辑
def run_all_experiments(seed_list, **kwargs):
    result_dir = kwargs.pop('result_dir')
    agg_num = kwargs['agg_num']
    repair_method = kwargs['repair_method']

    all_runs = []

    # ----------------------------------------------------
    # 您现在有两个选择：
    # A. 串行运行不同 seed 的实验，但每个 seed 内部是并行的 (推荐)
    for seed in seed_list:
        print(f"\n======== 开始运行 Seed {seed} 实验 ========")

        # 调用新的并行函数
        df = solve_lp_files_gurobi_parallel(seed=seed, **kwargs)

        filename = result_dir + f"/random_aggnum_{agg_num}_seed_{seed}_repair_{repair_method}.csv"
        df.to_csv(filename, index=False)
        print(f"数据已保存到: {filename}")
        all_runs.append(df)

    # B. (如果需要) 进一步并行化顶层的 seed 循环，但需谨慎避免 CPU 过载。
    # ----------------------------------------------------

    return all_runs





if __name__ == '__main__':

    problem = "CA"
    json_file_path = 'parameters/param_3.json'
    params = utils.get_problem_parameters(json_file_path)
    post_fix = utils.get_post_fix(params[problem])

    data_dir = "_".join([problem,post_fix])
    dataset_name = "test"
    lp_files_dir = f"./instance/{dataset_name}/{data_dir}"
    seed_list = [2,3,4,5,6]
    solve_num = 10
    agg_num = 50

    Threads = 0 # default 0
    # Threads = 2

    # agg_model_solve_time = -1 # 不限制时间
    agg_model_solve_time = 2
    time_limit = 1000
    neighborhood = {"k0":100,"k1":100,"Delta":5}

    # repair_method_list = ["None", "gurobi", "naive", "score", "subproblem","lightmilp"]
    repair_method_list = ["subproblem"]

    for repair_method in repair_method_list:

        result_dir = f"./result/random_{data_dir}_solve_{solve_num}_aggNum_{agg_num}_aggsolvetime_{agg_model_solve_time}_repair_{repair_method}_Threads_{Threads}"

        # 保存统计结果
        all_runs = []
        os.makedirs(result_dir, exist_ok=True)

        # 读取cache
        cache_dir = f"./cache/{dataset_name}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{data_dir}_solving_cache_threads_{Threads}.json')

        ## 单进程求解
        cache = utils.load_optimal_cache(cache_file, data_dir, lp_files_dir, solve_num, Threads,time_limit)
        # 每个seed，求解一次
        for seed in seed_list:
            random.seed(seed)
            df = accelerated_solving(cache, lp_files_dir, solve_num, agg_num, seed, problem, repair_method, agg_model_solve_time, time_limit,neighborhood,Threads)
            df.to_csv(result_dir + f"/random_aggnum_{agg_num}_seed_{seed}_repair_{repair_method}.csv", index=False)
            all_runs.append(df)

        ## 多进程求解
        """
        existing_cache = utils.load_cache(cache_file,data_dir)
        final_cache = utils.get_solving_cache_parallel(
            cache=existing_cache,
            cache_file=cache_file,
            directory=lp_files_dir,
            num_problems=solve_num,
            threads_per_solve=Threads,
            num_parallel_solves=10,
            time_limit=time_limit
        )

        # 配置实验参数
        experiment_params = {
            'cache': final_cache,
            'directory': lp_files_dir,
            'num_problems': solve_num,  # 仅运行一个模拟问题
            'agg_num': agg_num,
            'problem': 'test_set',
            'repair_method': repair_method,
            'PostSolve': True,
            'agg_model_solve_time': agg_model_solve_time,
            'time_limit': time_limit,
            # 新增的并行和线程参数
            'threads_per_solve': 2,  # 每个实例 Gurobi 使用 2 个线程
            'num_parallel_solves': 10,  # 同时求解 10 个实例
            # 修复和后处理的缺失参数
            'k0': 10,
            'k1': 10,
            'Delta': 5,
            'result_dir': result_dir
        }
        
        # 运行实验
        all_results = run_all_experiments(seed_list, **experiment_params)
        all_runs = all_results
        """

        # 合并所有 seed 的明细
        df_all = pd.concat(all_runs, ignore_index=True)
        df_all.to_csv(os.path.join(result_dir, "all_details.csv"), index=False)

        # 按 filename 汇总统计
        if repair_method != "None":
            # 修复，primal gap
            summary = df_all.groupby("filename").agg({
                "primal_gap": ["min", "max", "mean"],
                "time_reduce": ["min", "max", "mean"],
                "cons_reduce_ratio": ["mean"],
                "time_orig": ["mean"],
                "time_agg": ["mean"],
                "obj_orig": ["mean"],
                "obj_agg": ["mean"]
            })
        else:
            # 不修复，dual gap
            summary = df_all.groupby("filename").agg({
                "dual_gap":    ["min","max","mean"],
                "time_reduce": ["min","max","mean"],
                "cons_reduce_ratio": ["min","max","mean"],
                "time_orig": ["mean"],
                "time_agg": ["mean"],
                "obj_orig": ["mean"],
                "obj_agg": ["mean"]
            })
        # 扁平化多级列名
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary.reset_index(inplace=True)
        summary.to_csv(os.path.join(result_dir, "surrogate_summary.csv"), index=False)

        print("明细和摘要已保存至：", result_dir)


