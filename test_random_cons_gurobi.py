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

def get_a_assignmeng_lp():
    # 利用随机数创建一个成本矩阵cost_matrix
    driver_num = job_num = 5
    cost_matrix = np.zeros((driver_num, job_num))
    print("利用numpy生成的成本矩阵(全零)为：\n", cost_matrix)
    for i in range(driver_num):
        for j in range(job_num):
            random.seed(i * 5 + j)
            cost_matrix[i][j] = round(random.random() * 10 + 5, 0)
    print("利用rd.random生成的新成本矩阵为：\n", cost_matrix)  # np.zeros()生成的类型是<class 'numpy.ndarray'>
    print(type(cost_matrix))

    # 建模并起名
    model = Model("分配问题模型")

    # 定义决策变量及类型
    x = [[[] for i in range(driver_num)] for j in range(job_num)]
    for i in range(driver_num):
        for j in range(job_num):
            x[i][j] = model.addVar(vtype=GRB.BINARY, name='x' + str(i + 1) + str(j + 1))

    # 目标
    obj = LinExpr(0)
    for i in range(driver_num):
        for j in range(job_num):
            obj.addTerms(cost_matrix[i][j], x[i][j])
    model.setObjective(obj, GRB.MINIMIZE)

    # 约束
    for i in range(driver_num):
        f = LinExpr(0)  # 定义一个线性表达式叫f
        for j in range(job_num):
            f.addTerms(1, x[i][j])  # 一行的01变量之和为1
        model.addConstr(f == 1, name="row" + str(i + 1))
    for j in range(driver_num):
        f = LinExpr(0)
        for i in range(job_num):
            f.addTerms(1, x[i][j])  # 一列的01变量之和为1

    model.write("test_lp.lp")


def gen_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % p != 0 for p in primes):
            primes.append(num)
        num += 1
    return primes

def solve_lp_files_gurobi(directory: str, num_problems: int, agg_num: int, delete_con:bool, seed:int,problem:str,repair:bool,find_one:bool):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # 生成乘子
    primes = gen_primes(agg_num)
    u_list = [math.log(p) for p in primes]
    # u_list = primes
    # todo:现在对于等式约束的指派问题也没用，是出问题了？
    results = []

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        lp_path = os.path.join(directory, lp_file)
        print(f"Processing {lp_file}")


        ## 原问题
        model_orig = gp.read(lp_path)
        obj_sense = model_orig.ModelSense
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

        # 只能sample出同类型约束
        # todo
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

        # todo:根据约束类型添加聚合约束
        if problem == "assignment":
            model_agg.addConstr(expr == agg_rhs, name="agg_constraint")
        elif problem == "CA":
            model_agg.addConstr(expr <= agg_rhs, name="agg_constraint")
        else:
            raise Exception("unknown problem type")
        model_agg.update()
        cons_num_agg = model_agg.NumConstrs

        # todo
        # model_agg.setParam('SolutionLimit', 1)
        model_agg.setParam("TimeLimit", 1)
        model_agg.optimize()
        after_agg_cons_num = len(model_agg.getConstrs())
        if repair:
            Vars = model_agg.getVars()
            # vaule_dict = {var.VarName:var.X for var in Vars}
            vaule_list = [var.X for var in Vars]
            repair_model = gp.read(lp_path)
            repair_Vars = repair_model.getVars()
            for idx in range(len(repair_Vars)):
                # 感觉用VarHintVal更好
                repair_Vars[idx].Start = vaule_list[idx]
                # repair_Vars[idx].VarHintVal  = vaule_list[idx]

            # 找到一个解
            if find_one:
                repair_model.setParam('SolutionLimit', 1)
            else:
                # 求到最优
                pass

            repair_model.optimize()
            t3 = time.perf_counter()
            status_agg = repair_model.Status
            obj_agg = repair_model.ObjVal
            time_agg = t3 - t2

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
                "original_cons": original_cons_num,
                "after_agg_cons": after_agg_cons_num,
                "seed": seed,
            })
        else:
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
                "after_cons_num": cons_num_agg,
                "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
                "time_orig": time_orig,
                "time_agg": time_agg,
                "time_reduce": time_reduce,
                "original_cons": original_cons_num,
                "after_agg_cons": after_agg_cons_num,
                "seed": seed,
            })



    df = pd.DataFrame(results)
    return df

if __name__ == '__main__':
    # data_dir = "assignment_size_50_minval_200_maxval_300"

    # problem = "assignment"
    problem = "CA"
    # data_dir = "CA_500_700"
    data_dir = "CA_500_600"

    lp_files_dir = f"./instance/test/{data_dir}"
    solve_num = 1
    agg_num = 50
    delete_con = True
    repair = True
    find_one = True
    result_dir = f"./result/{data_dir}_agg_num_{agg_num}_delete_con_{str(delete_con)}_repair_{repair}_find_one_{find_one}"
    # 第一个是20的约束
    # 第二个490，取100吧


    # seed_list = [1,2,3,4,5,6]
    seed_list = [1]
    all_runs = []
    gurobi_solve = True
    os.makedirs(result_dir, exist_ok=True)
    for seed in seed_list:
        random.seed(seed)
        if gurobi_solve:
            df = solve_lp_files_gurobi(lp_files_dir, solve_num, agg_num,delete_con,seed,problem,repair,find_one)
            df.to_csv(result_dir + f"/surrogate_stats_aggnum_{agg_num}_seed_{seed}_repair_{repair}.csv", index=False)
            all_runs.append(df)

    # 合并所有 seed 的明细
    df_all = pd.concat(all_runs, ignore_index=True)
    df_all.to_csv(os.path.join(result_dir, "all_details.csv"), index=False)
    # 按 filename 汇总统计
    if repair:
        summary = df_all.groupby("filename").agg({
            "primal_gap": ["min", "max", "mean"],
            "time_reduce": ["min", "max", "mean"],
            "cons_reduce_ratio": ["min", "max", "mean"]
        })
    else:
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


