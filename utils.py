import numpy as np
import random
from gurobipy import Model, GRB, LinExpr
import os
import json
import gurobipy as gp
import time
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

def load_cache(cache_dir,data_dir,lp_files_dir,solve_num,Threads):

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