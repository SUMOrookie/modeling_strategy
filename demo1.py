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

def accelerated_solving(cache:dict, directory: str, num_problems: int, agg_num: int, seed:int, problem:str, repair_method:str,
                        agg_model_solve_time, time_limit,neighborhood,Threads,post_solve_method
                        ,KEEP_FRACTION,NORM_METHOD,SIMILARITY_METRIC):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # 生成乘子
    primes = utils.gen_primes(agg_num)
    # u_list = [math.log(p) for p in primes]
    u_list = [1 for p in primes]

    results = []
    evert_time_primal_gap = []

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        # 获取lp路径
        lp_path = os.path.join(directory, lp_file)
        print(f"Processing {lp_file}")

        #### 原问题求解
        # 如果缓存中已有结果，就直接读取，否则报错
        if lp_path in cache:
            print("------------read cache-------------")
            entry = cache[lp_path]
            obj_sense = entry['obj_sense']
            status_orig = entry['status_orig']
            bks = entry['obj_orig']
            time_orig = entry['time_orig']
            obj_at_1000 = entry['obj_at_1000']
        else:
            raise Exception("there is not cache")

        #### 聚合问题求解
        # 读入问题
        model_agg = gp.read(lp_path)
        t0 = time.perf_counter()
        cons_num_orig = model_agg.NumConstrs

        model_relax = model_agg.copy()
        # 聚合
        # utils.aggregate_constr(model_agg,agg_num)
        # utils.aggregate_constr_two_two(model_agg,agg_num)
        # utils.aggregate_constr_duals(model_agg,KEEP_FRACTION,NORM_METHOD,SIMILARITY_METRIC)
        cons_num_agg = model_agg.NumConstrs

        # 求解聚合问题
        print("------------solving agg model-------------")
        model_agg.setParam("Threads",Threads)
        model_agg.setParam("Seed", seed)
        if agg_model_solve_time == -1:
            model_agg.optimize()
        else:
            model_agg.setParam("TimeLimit", 100)
            model_agg.optimize()

        solution = {v.VarName:v.X for v in model_agg.getVars()}
        # 松弛andfix
        for v in model_relax.getVars():
            v.LB = solution[v.VarName]
            v.UB = solution[v.VarName]
            v.VType = GRB.CONTINUOUS
        model_relax.optimize()
        dual_values = [c.Pi for c in model_relax.getConstrs()]
        # 4. 获取对偶值 (现在可以获取了！)
        # dual_values = [c.Pi for c in fixed_model.getConstrs()]
        # x = [v.X for v in model_agg.getVars()]
        # dual_values = [c.Pi for c in model_agg.getConstrs()]
        # print("dual_values:",dual_values)
        # # 备份，用于固定变量后的求解
        # model_agg_2 = model_agg.copy()
        #
        # # 松弛
        # for v in model_agg.getVars():
        #     try:
        #         v.VType = GRB.CONTINUOUS
        #     except Exception:
        #         # 如果直接赋值失败，忽略（继续）
        #         pass
        # model_agg.update()
        #
        # # 求解聚合问题
        # print("------------solving agg model-------------")
        # model_agg.setParam("Threads",Threads)
        # model_agg.setParam("Seed", seed)
        # if agg_model_solve_time == -1:
        #     model_agg.optimize()
        # else:
        #     model_agg.setParam("TimeLimit", agg_model_solve_time)
        #     model_agg.optimize()
        #
        # # 获得松弛解
        # Vars = model_agg.getVars()
        # vaule_dict = {var.VarName: var.X for var in Vars}
        #
        #
        # dual_values = [c.Pi for c in model_agg.getConstrs()]
        # reduced_costs = [v.RC for v in model_agg.getVars()]
        # mean_less_than_zero = sum([r if r < 0 else 0 for r in reduced_costs])/sum([1 if r < 0 else 0 for r in reduced_costs])
        # fixed_vars = {v.VarName:0 for v in model_agg.getVars() if v.RC  <= mean_less_than_zero}
        #
        # # 把那些小于平均值的变量给固定为0，然后相当于得到了部分变量取值
        # # 那么对于其他变量，这些变量是小数，是不是可以恢复限制然后求解一下。
        # # 构造一个子问题（部分变量固定为0），然后再用来求解松弛模型？好绕。
        # for v in model_agg_2.getVars():
        #     if v.VarName in fixed_vars:
        #         v.LB = fixed_vars[v.VarName]
        #         v.UB = fixed_vars[v.VarName]
        # model_agg_2.setParam("Threads",Threads)
        # model_agg_2.setParam("Seed", seed+1)
        # model_agg_2.setParam("TimeLimit", agg_model_solve_time)
        # model_agg_2.optimize()
        # # 获得解
        # Vars = model_agg_2.getVars()
        # vaule_dict = {var.VarName: var.X for var in Vars}

        # 松弛解可行性修复
        repair_model = gp.read(lp_path)
        repair_model.setParam("Threads", Threads)
        repair_model.setParam("Seed", seed+1)
        vaule_dict = repair_and_post_solve_func.repair(repair_model,vaule_dict,repair_method,lp_path)

        #### 邻域搜索（保证最优性）
        t_before_search = time.perf_counter()
        ## 加bound
        bound = 0
        for varname, val in vaule_dict.items():
            bound += repair_model.getVarByName(varname).Obj * val
        print("bound:", bound)
        repair_model.params.Cutoff = bound
        metrics_list = repair_and_post_solve_func.PostSolve(repair_model,neighborhood,vaule_dict,lp_file,t0,time_limit,post_solve_method)
        t1 = time.perf_counter()

        ### 计算指标
        status_agg = repair_model.Status
        obj_agg = repair_model.ObjVal
        total_time_agg = t1 - t0


        # 得到每一个时刻的primal_gap
        # trajectory = metrics_list[0]['trajectory'] # alist,元素为{'gap': 4.459429741595124, 'obj': 39290.92221908511, 'time': 0}
        # base_time = round(t_before_search-t0)
        # for item in trajectory:
        #     obj = item['obj']
        #     if obj_sense == GRB.MINIMIZE:
        #         primal_gap = (obj - bks) / abs(bks) if bks != 0 else float("inf")
        #     else:
        #         primal_gap = (bks - obj) / abs(bks) if bks != 0 else float("inf")
        #     item['primal_gap'] = primal_gap
        #     item['time'] += base_time

        ## 时间约简计算
        time_reduce = (time_orig - total_time_agg) / time_orig if time_orig > 0 else 0

        # if obj_sense == GRB.MINIMIZE:
        #     print("最小化问题")
        #     primal_gap = (obj_agg - bks) / abs(bks) if bks != 0 else float("inf")
        # else:
        #     print("最大化问题")
        #     primal_gap = (bks - obj_agg) / abs(bks) if bks != 0 else float("inf")


        if obj_at_1000 is not None:
            print("达到时间限制")
            if obj_sense == GRB.MINIMIZE:
                solver_primal_gap = obj_at_1000 - bks
                our_primal_gap = obj_agg - bks
            else:
                solver_primal_gap = bks - obj_at_1000
                our_primal_gap = bks - obj_agg
            print(f"bks:{bks}")
            print(f"求解器obj:{obj_at_1000}\t 聚合后obj:{obj_agg}")
            print(f"求解器primal_gap:{solver_primal_gap},\t 聚合后primal_gap:{our_primal_gap}")
            if abs(solver_primal_gap) > 1e-4:
                gain = (solver_primal_gap-our_primal_gap)/solver_primal_gap * 100
            elif abs(our_primal_gap) > 1e-4:
                gain = 100
            else:
                gain = None
            print(f"gain:{gain}")
        else:
            print("求解器1000s内求解完成")
            print(f"bks:{bks}")
            solver_primal_gap = 0
            if obj_sense == GRB.MINIMIZE:
                our_primal_gap = obj_agg - bks
            else:
                our_primal_gap = bks - obj_agg
            gain = None
            print(f"原obj:{bks},\t 聚合后obj：{obj_agg}")
            print(f"原时间:{time_orig},\t 聚合后时间:{total_time_agg}")
            print(f"求解器primal_gap:{solver_primal_gap}")
            print(f"聚合后primal_gap:{our_primal_gap}")
            print(f"time_reduce:{time_reduce}")

        ## 保存
        # evert_time_primal_gap.append(trajectory)
        results.append({
            "filename": lp_file,
            "status_orig": status_orig,
            "status_agg": status_agg,
            "obj_orig": bks,
            "obj_at_1000":obj_at_1000,
            "obj_agg": obj_agg,
            "our_primal_gap": our_primal_gap,
            "solver_primal_gap": solver_primal_gap,
            "gain":gain,
            "original_cons_num": cons_num_orig,
            "after_cons_num": cons_num_agg,
            "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
            "time_orig": time_orig,
            "time_agg": total_time_agg,
            "time_reduce": time_reduce,
            "original_cons": cons_num_orig,
            "after_agg_cons": cons_num_agg,
        })

    df = pd.DataFrame(results)
    return df,evert_time_primal_gap





if __name__ == '__main__':

    problem = "CA"
    json_file_path = 'parameters/param_0.json'
    params = utils.get_problem_parameters(json_file_path)
    post_fix = utils.get_post_fix(params[problem])

    data_dir = "_".join([problem, post_fix])
    dataset_name = "test"
    lp_files_dir = f"./instance/{dataset_name}/{data_dir}"
    # seed_list = [2, 3, 4, 5, 6]
    seed_list = [2]
    solve_num = 10
    agg_num = 50
    Threads = 0  # default 0

    # agg_model_solve_time = -1 # 不限制时间
    agg_model_solve_time = 2
    time_limit = 1000
    neighborhood = {"k0": 300, "k1": 100, "Delta": 100}
    neighborhood_str = "_".join([key + "_" + str(val) for key, val in neighborhood.items()])
    # repair_method_list = ["None", "gurobi", "naive", "score", "subproblem","lightmilp"]
    repair_method_list = ["subproblem"]
    # repair_method_list = ["subproblem_cons_geq"]
    # repair_method_list = ["None"]
    post_solve_method = "neighborhood"
    dual = False
    KEEP_FRACTION = 0.8
    NORM_METHOD = "l2"
    SIMILARITY_METRIC = "jaccard"

    for repair_method in repair_method_list:
        if dual:
            result_dir = f"./result/random_{KEEP_FRACTION}_{NORM_METHOD}_{SIMILARITY_METRIC}_{data_dir}_solve_{solve_num}_aggsolvetime_{agg_model_solve_time}_repair_{repair_method}_{post_solve_method}_{neighborhood_str}"
        else:
            result_dir = f"./result/random_{data_dir}_solve_{solve_num}_aggNum_{agg_num}_aggsolvetime_{agg_model_solve_time}_repair_{repair_method}_{post_solve_method}_{neighborhood_str}"

        # 保存统计结果
        all_runs = []
        os.makedirs(result_dir, exist_ok=True)

        # 读取cache
        # cache_dir = f"./cache/{dataset_name}"
        # os.makedirs(cache_dir, exist_ok=True)
        # cache_file_path = os.path.join(cache_dir, f'{data_dir}_solving_cache_threads_{Threads}.json')
        cache_file_path = utils.get_cache_file_path(dataset_name, data_dir)

        ## 单进程求解
        cache = utils.load_optimal_cache(cache_file_path, lp_files_dir, solve_num, Threads, 3600)
        # 每个seed，求解一次
        for seed in seed_list:
            random.seed(seed)
            df, every_second_gap_agg = accelerated_solving(cache, lp_files_dir, solve_num, agg_num, seed, problem,
                                                           repair_method,
                                                           agg_model_solve_time, time_limit, neighborhood, Threads,
                                                           post_solve_method
                                                           , KEEP_FRACTION, NORM_METHOD, SIMILARITY_METRIC)
            df.to_csv(
                result_dir + f"/random_aggnum_{agg_num}_seed_{seed}_repair_{repair_method}_{post_solve_method}.csv",
                index=False)
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
                "our_primal_gap": ["min", "max", "mean"],
                "solver_primal_gap": ["min", "max", "mean"],
                "gain": ["min", "max", "mean"],
                "time_reduce": ["min", "max", "mean"],
                "cons_reduce_ratio": ["mean"],
                "time_orig": ["mean"],
                "time_agg": ["mean"],
                "obj_orig": ["mean"],
                "obj_at_1000": ["mean"],
                "obj_agg": ["mean"]
            })
        else:
            # 不修复，dual gap
            summary = df_all.groupby("filename").agg({
                "dual_gap": ["min", "max", "mean"],
                "time_reduce": ["min", "max", "mean"],
                "cons_reduce_ratio": ["min", "max", "mean"],
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

