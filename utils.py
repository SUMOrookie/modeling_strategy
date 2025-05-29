import numpy as np
import random
from gurobipy import Model, GRB, LinExpr
import os
import json
import gurobipy as gp
import time
import utils
import math


# 全局／外部列表，用来存所有 solve 的 metrics
all_metrics = []
def make_callback(solve_id, metrics_list):
    """返回一个回调函数，该回调会把当前 solve 的时间和 gap 存到 metrics_list."""
    info = {"solve_id": solve_id,
            "hit": False,        # 是否已经记录过
            "time_at_1pct": None,
            "gap_at_hit": None,
            "time_perf_counter":None}

    metrics_list.append(info)

    start = time.perf_counter()
    def cb(model, where):
        if where == GRB.Callback.MIP and not info["hit"]:
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            if abs(best) > 1e-6:
                gap = abs(best - bound) / abs(best)
            else:
                gap = float("inf")
            if gap <= 0.01:
                info["hit"] = True
                now = time.perf_counter()
                info["time_perf_counter"] = now
                info["time_at_1pct"] = now - start
                info["gap_at_hit"]  = gap

    return cb

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
def decimal_to_binary_list(n, i):
    """
    将十进制整数 i 转换为二进制列表，确保列表长度与 n-1 的二进制位数相同。

    参数:
    n (int): 用于确定二进制位数的上限值（生成的二进制位数与 n-1 的位数相同）
    i (int): 需要转换的十进制整数

    返回:
    list: 包含二进制字符的列表，长度与 n-1 的二进制位数相同
    """
    if n <= 0:
        raise ValueError("n 必须是正整数")

    # 计算所需的位数（即 n-1 的二进制位数）
    max_bits = len(bin(n - 1)) - 2  # 减2是因为bin()返回的字符串前缀是 '0b'

    # 将 i 转换为指定位数的二进制字符串，并拆分为列表
    return [int(c) for c in format(i, f'0{max_bits}b')]

def z_score_normalize(lst):
    if not lst:
        return []
    mean = sum(lst) / len(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    std_dev = variance ** 0.5
    if std_dev == 0:  # 处理所有元素相同的情况
        return [0.0] * len(lst)
    return [(x - mean) / std_dev for x in lst]


def gen_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % p != 0 for p in primes):
            primes.append(num)
        num += 1
    return primes


def get_solving_cache(cache:dict,cache_file:str,directory: str, num_problems: int,Threads:int):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        # 得到lp路径
        lp_path = os.path.join(directory, lp_file)
        print(f"Processing {lp_file} cache")

        ## 原问题求解
        # 如果缓存中已有结果，就直接读取，否则求解并写入缓存
        if lp_path in cache:
            pass
        else:
            print("------------there is not cache, solving-------------")
            # 读入模型
            model_orig = gp.read(lp_path)

            # 时间从读入开始算,求解
            model_orig.setParam("Threads", Threads)
            t0 = time.perf_counter()
            model_orig.optimize()
            t1 = time.perf_counter()

            # 指标
            obj_sense = model_orig.ModelSense
            status_orig = model_orig.Status
            obj_orig = model_orig.ObjVal
            time_orig = t1 - t0

            # 写入缓存
            cache[lp_path] = {
                'obj_sense':   obj_sense,
                'status_orig': status_orig,
                'obj_orig':    obj_orig,
                'time_orig':   time_orig
            }
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)

def get_gap_cache(cache,cache_file,lp_dir_path, solve_num,Threads):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:solve_num]

    # 依次读取并求解每个 .lp 文件
    for lp_file in lp_files:
        # 得到lp路径
        lp_path = os.path.join(lp_dir_path, lp_file)
        print(f"Processing {lp_file} cache")

        ## 原问题求解
        # 如果缓存中已有结果，就直接读取，否则求解并写入缓存
        if lp_path in cache:
            pass
        else:
            print("------------there is not cache, solving-------------")
            # 读入模型
            model_orig = gp.read(lp_path)

            # 时间从读入开始算,求解
            model_orig.setParam("Threads", Threads)
            # model_orig.setParam("MIPGap", 1e-2)
            t0 = time.perf_counter()

            # 记录是否已经输出过信息
            cb = make_callback(lp_file, all_metrics)
            model_orig.optimize(cb)
            t1 = time.perf_counter()

            # 指标
            obj_sense = model_orig.ModelSense
            status_orig = model_orig.Status
            obj_orig = model_orig.ObjVal
            time_gap1_orig = t1 - t0

            # 写入缓存
            cache[lp_path] = {
                'obj_sense':   obj_sense,
                'status_orig': status_orig,
                'obj_orig':    obj_orig,
                'time_orig':   time_gap1_orig,
                'gap_at_hit_1pct':all_metrics[-1]['gap_at_hit'],
                'hit':all_metrics[-1]['hit'],
                'time_at_1pct':all_metrics[-1]['time_at_1pct']
            }
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
def load_gap_cache(cache_dir, task_name, lp_dir_path, solve_num, Threads):
    os.makedirs(cache_dir,exist_ok=True)
    cache_file = os.path.join(cache_dir,f'{task_name}_solve_gap_cache.json')

    # 加载缓存（如果存在）
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    # utils.get_solving_cache(cache,cache_file,lp_dir_path, solve_num,Threads)
    get_gap_cache(cache,cache_file,lp_dir_path, solve_num,Threads)
    return cache

def load_optimal_cache(cache_dir, data_dir, lp_files_dir, solve_num, Threads=0):

    os.makedirs(cache_dir,exist_ok=True)
    cache_file = os.path.join(cache_dir,f'{data_dir}_solve_cache.json')

    # 加载缓存（如果存在）
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    get_solving_cache(cache,cache_file,lp_files_dir, solve_num,Threads)
    return cache


def generate_and_save_feasible_model(lp_path, out_dir,
                                     initial_rhs=1,
                                     initial_frac=0.5,
                                     seed=None):
    """
    1. 读取原模型；
    2. 迭代添加随机 ≥ 初始RHS的新约束，数量为原约束数 * initial_frac；
       - 若当前新模型不可行，则将约束数量减半重试；
       - 若可行，则将模型写入 out_dir/"new_constr" 下，并在文件名加上 "new_constr"。
    """
    if seed is not None:
        random.seed(seed)
        # 读入原模型
    model0 = gp.read(lp_path)
    orig_constrs = model0.getConstrs()
    orig_count = len(orig_constrs) // 5


    # 计算初始要添加的约束数
    num_new = max(1, int(orig_count * initial_frac))

    # 确保输出目录存在
    save_dir = out_dir + "_new_constr"
    os.makedirs(save_dir, exist_ok=True)

    iteration = 0
    while num_new >= 1:
        iteration += 1
        # 深拷贝原模型
        model = model0.copy()
        all_vars = model.getVars()
        # 添加 num_new 条随机 ≥ 约束
        for i in range(num_new):
            # 随机选变量个数 k
            k = random.randint(2, 10)
            vars_in_expr = random.sample(all_vars, k)
            expr = gp.quicksum(vars_in_expr)
            model.addConstr(expr >= initial_rhs, name=f"rand_ge_{iteration}_{i}")
        model.update()

        # 判断可行性
        model.Params.OutputFlag = 0  # 关闭求解器日志
        model.Params.SolutionLimit = 1
        model.optimize()
        status = model.Status

        if status == gp.GRB.SOLUTION_LIMIT:
            print("已找到可行解，提前终止")
            Vars = model.getVars()
            # for var in Vars:
            #     print(var.VarName,"\t",var.X)
            # 可行，则保存模型并结束
            base_name = os.path.splitext(os.path.basename(lp_path))[0]
            save_path = os.path.join(
                save_dir,
                f"{base_name}_new_constr_{num_new}.lp"
            )
            model.write(save_path)
            print(f"[迭代{iteration}] 可行模型已保存：{save_path}")
            return save_path
        else:
            # 不可行，约束数减半，重试
            print(f"[迭代{iteration}] 不可行，约束数 {num_new} -> {num_new // 2}")
            num_new //= 2

    raise RuntimeError("无法通过随机添加 ≥ 约束获得可行解；所有尝试均失败。")

def aggregate_constr(model_agg,agg_num=None,sample=None):
    # 对于sample出的约束，要分为大于等于、小于等于和等于
    conss = model_agg.getConstrs()

    if sample == None:
        sample = random.sample(conss, min(agg_num, len(conss)))
    if agg_num == None:
        agg_num = 50

    # 乘子
    # primes = utils.gen_primes(agg_num)
    # u_list = [math.log(p) for p in primes]

    u_list = [1 for i in range(agg_num)]
    # 计算聚合约束
    agg_coeffs_leq = {}
    agg_rhs_leq = 0.0
    agg_coeffs_geq = {}
    agg_rhs_geq = 0.0
    agg_coeffs_eq = {}
    agg_rhs_eq = 0.0
    for idx, cons in enumerate(sample):
        u = u_list[idx]
        constr_expr = model_agg.getRow(cons)
        sense = cons.Sense
        for j in range(constr_expr.size()):
            var = constr_expr.getVar(j)
            coef = constr_expr.getCoeff(j)
            if sense == "<":
                agg_coeffs_leq[var.VarName] = agg_coeffs_leq.get(var.VarName, 0.0) + u * coef
            elif sense == ">":
                agg_coeffs_geq[var.VarName] = agg_coeffs_geq.get(var.VarName, 0.0) + u * coef
            elif sense == "=":
                agg_coeffs_eq[var.VarName] = agg_coeffs_eq.get(var.VarName, 0.0) + u * coef
            else:
                raise Exception("unknown constr sense")
        if sense == "<":
            agg_rhs_leq += u * cons.RHS
        elif sense == ">":
            agg_rhs_geq += u * cons.RHS
        elif sense == "=":
            agg_rhs_eq += u * cons.RHS
        else:
            raise Exception("unknown constr sense")
        model_agg.remove(cons)  # 删除约束
    model_agg.update()

    # 构造聚合约束
    expr_leq = 0
    expr_geq = 0
    expr_eq = 0
    for var_name, coef in agg_coeffs_leq.items():
        var = model_agg.getVarByName(var_name)
        expr_leq += coef * var
    for var_name, coef in agg_coeffs_geq.items():
        var = model_agg.getVarByName(var_name)
        expr_geq += coef * var
    for var_name, coef in agg_coeffs_eq.items():
        var = model_agg.getVarByName(var_name)
        expr_eq += coef * var

    model_agg.addConstr(expr_leq <= agg_rhs_leq, name="agg_constraint_leq")
    model_agg.addConstr(expr_geq >= agg_rhs_geq, name="agg_constraint_geq")
    model_agg.addConstr(expr_eq == agg_rhs_eq, name="agg_constraint_eq")
    model_agg.update()