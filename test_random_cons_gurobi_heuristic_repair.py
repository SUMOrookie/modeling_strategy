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
import repair_func
import utils

def solve_lp_files_gurobi(cache:dict,directory: str, num_problems: int, agg_num: int,seed:int,problem:str,repair_method:str,PostSolve:bool,agg_model_solve_time):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致


    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # 生成乘子
    primes = utils.gen_primes(agg_num)
    u_list = [math.log(p) for p in primes]

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
            # print("------------there is not cache, solving-------------")
            # # 读入模型
            # model_orig = gp.read(lp_path)
            #
            # # 时间从读入开始算,求解
            # t0 = time.perf_counter()
            # model_orig.optimize()
            # t1 = time.perf_counter()
            #
            # # 指标
            # obj_sense = model_orig.ModelSense
            # status_orig = model_orig.Status
            # obj_orig = model_orig.ObjVal
            # time_orig = t1 - t0
            #
            # # 写入缓存
            # cache[lp_path] = {
            #     'obj_sense':   obj_sense,
            #     'status_orig': status_orig,
            #     'obj_orig':    obj_orig,
            #     'time_orig':   time_orig
            # }
            # with open(cache_file, 'w') as f:
            #     json.dump(cache, f, indent=2)

        ## 聚合问题求解
        # 读入问题
        model_agg = gp.read(lp_path)
        t2 = time.perf_counter()

        # 指标
        cons_num_orig = model_agg.NumConstrs

        # 聚合
        utils.aggregate_constr(model_agg,agg_num)

        #
        #
        # # 只能sample出同类型约束,todo
        # conss = model_agg.getConstrs()
        # sample = random.sample(conss, min(agg_num, len(conss)))
        #
        # # 计算聚合约束
        # agg_coeffs = {}
        # agg_rhs = 0.0
        # for idx, cons in enumerate(sample):
        #     u = u_list[idx]
        #     constr_expr = model_agg.getRow(cons)
        #     for j in range(constr_expr.size()):
        #         var = constr_expr.getVar(j)
        #         coef = constr_expr.getCoeff(j)
        #         agg_coeffs[var.VarName] = agg_coeffs.get(var.VarName, 0.0) + u * coef
        #     agg_rhs += u * cons.RHS
        #     model_agg.remove(cons) # 删除约束
        # model_agg.update()
        #
        # # 构造聚合约束
        # expr = 0
        # for var_name, coef in agg_coeffs.items():
        #     var = model_agg.getVarByName(var_name)
        #     expr += coef * var
        #
        # # todo:根据约束类型定义符号
        # if problem == "assignment":
        #     model_agg.addConstr(expr == agg_rhs, name="agg_constraint")
        # elif problem == "CA":
        #     model_agg.addConstr(expr <= agg_rhs, name="agg_constraint")
        # else:
        #     raise Exception("unknown problem type")
        # model_agg.update()

        # 聚合后约束数量
        cons_num_agg = model_agg.NumConstrs

        ## 一些求解的参数 todo
        # 解数量，时间限制
        # model_agg.setParam('SolutionLimit', 1)
        print("------------solving agg model-------------")
        model_agg.setParam("Threads",Threads)
        if agg_model_solve_time == -1:
            model_agg.optimize()
        else:
            model_agg.setParam("TimeLimit", agg_model_solve_time)
            model_agg.optimize()
        agg_objval_original = model_agg.ObjVal

        if repair_method == "None":
            # 不修复
            print("------------do not repair-------------")
            t3 = time.perf_counter()
            status_agg = model_agg.Status
            obj_agg = model_agg.ObjVal
            time_agg = t3 - t2

            after_agg_cons_num = len(model_agg.getConstrs())

            if obj_sense == GRB.MINIMIZE:
                print("最小化问题")
                dual_gap = (obj_orig - obj_agg) / abs(obj_orig) if obj_orig != 0 else float("inf")
            else:
                print("最大化问题")
                dual_gap = (obj_agg - obj_orig) / abs(obj_orig) if obj_orig != 0 else float("inf")
            time_reduce = (time_orig - time_agg) / time_orig if time_orig > 0 else 0

            print(f"原obj:{obj_orig},\t 聚合后obj：{obj_agg}")
            print(f"原时间:{time_orig},\t 聚合后时间:{time_agg}")
            print(f"dual gap:{dual_gap}")
            print(f"time_reduce:{time_reduce}")

            results.append({
                "filename": lp_file,
                "status_orig": status_orig,
                "status_agg": status_agg,
                "obj_orig": obj_orig,
                "obj_agg": obj_agg,
                "dual_gap": dual_gap,
                "original_cons_num": cons_num_orig,
                "after_cons_num": cons_num_agg,
                "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
                "time_orig": time_orig,
                "time_agg": time_agg,
                "time_reduce": time_reduce,
                "original_cons": cons_num_orig,
                "after_agg_cons": cons_num_agg,
                "seed": seed,
            })
        else:
            # 获得变量值
            Vars = model_agg.getVars()
            vaule_dict = {var.VarName:var.X for var in Vars}

            # 读入新模型，用于修复
            repair_model = gp.read(lp_path)
            repair_model.setParam("Threads",Threads)

            # todo：gurobi修复，并且求到最优解与否
            # 如果gurobi and postsolve：不设置SolutionLimit
            # 如果gurobi and not postsolve：设置SolutionLimit
            # 其他情况，调用对应的启发式函数
            # 然后判断postsolve
            if repair_method == "gurobi":
                if PostSolve:
                    pass
                else:
                    # gurobi修复
                    repair_model.setParam('SolutionLimit', 1)
            else:
                if repair_method == "naive":
                # 简单的启发式修复
                    vaule_dict = repair_func.heuristic_repair(repair_model, vaule_dict)
                elif repair_method == "score":
                # 变量评分
                    vaule_dict = repair_func.heuristic_repair_with_score(repair_model, vaule_dict)
                elif repair_method == "subproblem":
                    vaule_dict = repair_func.heuristic_repair_subproblem(repair_model, vaule_dict)
                elif repair_method == "lightmilp":
                    vaule_dict = repair_func.heuristic_repair_light_MILP(repair_model,vaule_dict,lp_path)
                else:
                    raise Exception("unknown repair_method")

            # postsolve的意思是，接入原模型，求解至最优解
            if PostSolve:
                ## 修复后，作为原模型初始解求解，也就是再接入求解器
                print("------------PostSolve-------------")
                # 赋初始值
                if repair_method != "gurobi":
                    repair_Vars = repair_model.getVars()
                    for idx in range(len(repair_Vars)):
                        varname = repair_Vars[idx].VarName
                        if vaule_dict[varname]!= None:
                            repair_model.getVarByName(varname).Start = vaule_dict[varname]
                        # repair_Vars[idx].VarHintVal  = vaule_list[idx]
                        # 修复后的作为start，松弛的用hintval

                # 求解（修复）,计算指标
                repair_model.optimize()

                t3 = time.perf_counter()
                status_agg = repair_model.Status
                obj_agg = repair_model.ObjVal
                time_agg = t3 - t2

                ## gap、时间约简计算
                if obj_sense == GRB.MINIMIZE:
                    print("最小化问题")
                    primal_gap = (obj_agg - obj_orig) / abs(obj_orig) if obj_orig != 0 else float("inf")
                else:
                    print("最大化问题")
                    primal_gap = (obj_orig - obj_agg) / abs(obj_orig) if obj_orig != 0 else float("inf")
                time_reduce = (time_orig - time_agg) / time_orig if time_orig > 0 else 0

                print(f"原obj:{obj_orig},\t 聚合后obj：{obj_agg}")
                print(f"原时间:{time_orig},\t 聚合后时间:{time_agg}")
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
                    "time_agg": time_agg,
                    "time_reduce": time_reduce,
                    "original_cons": cons_num_orig,
                    "after_agg_cons": cons_num_agg,
                    "seed": seed,
                })
            
            else:
                # 这里只是算一下obj值，不会再接入求解器了
                print("------------do no PostSolve-------------")
                if repair_method == "gurobi":
                    # 不后处理，也就是得到一个解就行
                    repair_model.optimize()
                t3 = time.perf_counter()
                status_agg =  model_agg.Status # 应该不需要了
                time_agg = t3 - t2

                # 获得目标值
                if repair_method != "gurobi":
                    obj_expr = model_agg.getObjective()
                    obj_value = 0.0
                    obj_value += obj_expr.getConstant()
                    var_in_constr = {obj_expr.getVar(idx).VarName: idx for idx in range(obj_expr.size())}
                    for var_name, var_val in vaule_dict.items():
                        if var_name in var_in_constr:
                            obj_value += obj_expr.getCoeff(var_in_constr[var_name]) * var_val
                else:
                    obj_value = repair_model.ObjVal

                if obj_sense == GRB.MINIMIZE:
                    print("最小化问题")
                    primal_gap = (obj_value - obj_orig) / abs(obj_orig) if obj_orig != 0 else float("inf")
                else:
                    print("最大化问题")
                    primal_gap = (obj_orig - obj_value) / abs(obj_orig) if obj_orig != 0 else float("inf")
                time_reduce = (time_orig - time_agg) / time_orig if time_orig > 0 else 0

                print(f"原obj:{obj_orig},\t 聚合后obj：{agg_objval_original},\t 修复后obj:{obj_value}")
                print(f"原时间:{time_orig},\t 聚合后时间:{time_agg}")
                print(f"primal gap:{primal_gap}")

                print(f"time_reduce:{time_reduce}")
                # 保存
                results.append({
                    "filename": lp_file,
                    "status_orig": status_orig,
                    "status_agg": status_agg,
                    "obj_orig": obj_orig,
                    "obj_agg_no_repair": agg_objval_original,
                    "obj_agg_after_repair": obj_value,
                    "primal_gap": primal_gap,
                    "original_cons_num": cons_num_orig,
                    "after_cons_num": cons_num_agg,
                    "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
                    "time_orig": time_orig,
                    "time_agg": time_agg,
                    "time_reduce": time_reduce,
                    "original_cons": cons_num_orig,
                    "after_agg_cons": cons_num_agg,
                    "seed": seed,
                })

    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    # data_dir = "assignment_size_50_minval_200_maxval_300"

    # problem = "assignment"
    problem = "CA"
    # data_dir = "CA_500_700"
    # data_dir = "CA_500_600"
    data_dir = "CA_500_600"
    # data_dir = "CA_200_400"

    # lp_files_dir = f"./instance/test/{data_dir}"
    lp_files_dir = f"./instance/test/{data_dir}"
    seed_list = [2,3,4,5,6]
    # seed_list = [2,3,4,5,6,7,8,9,10,11]
    # seed_list = [1]
    solve_num = 10
    agg_num = 50
    Threads = 0 # default 0
    # Threads = 5
    # agg_model_solve_time = -1 # 不限制时间
    agg_model_solve_time = 1

    repair_method_list = ["None", "gurobi", "naive", "score", "subproblem","lightmilp"]
    post_solve = True
    for idx in range(5,6):

        repair_method = repair_method_list[idx]

        result_dir = f"./result/{data_dir}_solve_{solve_num}_aggNum_{agg_num}_aggsolvetime_{agg_model_solve_time}_repair_{repair_method}_PostSolve_{post_solve}_Threads_{Threads}"

        # 保存统计结果
        all_runs = []
        os.makedirs(result_dir, exist_ok=True)

        # 读取cache
        cache_dir = "./cache"
        cache = utils.load_cache(cache_dir,data_dir,lp_files_dir,solve_num,Threads)
        #
        # 每个seed，求解一次
        for seed in seed_list:
            random.seed(seed)
            df = solve_lp_files_gurobi(cache, lp_files_dir, solve_num, agg_num, seed, problem, repair_method, post_solve, agg_model_solve_time)
            df.to_csv(result_dir + f"/surrogate_stats_aggnum_{agg_num}_seed_{seed}_repair_{repair_method}.csv", index=False)
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


