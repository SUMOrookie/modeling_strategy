import numpy as np
import random
from gurobipy import Model, GRB, LinExpr
import os
import json
import gurobipy as gp
import time
import concurrent.futures
import utils
import math


# # 全局／外部列表，用来存所有 solve 的 metrics
all_metrics = []
def make_callback(solve_id, metrics_list):
    """返回一个回调函数，该回调会把当前 solve 的时间和 gap 存到 metrics_list."""
    info = {"solve_id": solve_id,
            "hit": False,        # 是否已经记录过达到1%gap
            "time_at_1pct": None,
            "gap_at_hit": None,
            "time_perf_counter":None,

            # 关于1000s时的数据
            "hit_1000": False,  # 是否已记录过 1000 秒状态
            "obj_at_1000": None,
            "gap_at_1000": None,

            # 每一秒的数据
            "trajectory": [], #  每一秒记录 {"time": ..., "obj": ..., "gap": ...}
            }

    metrics_list.append(info)
    last_recorded_second = {"value": -1}
    start = time.perf_counter()
    def cb(model, where):
        if where == GRB.Callback.MIP and not info["hit"]:
            # now = time.perf_counter()
            # 上面的计时方式是遗留问题

            elapsed = model.cbGet(GRB.Callback.RUNTIME)
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

            # 记录 1000 秒时的目标值
            if elapsed >= 1000 and not info["hit_1000"]:
                info["hit_1000"] = True
                info["obj_at_1000"] = best
                info["gap_at_1000"] = gap

            #  每整秒记录一次 obj/gap
            current_second = int(elapsed)
            if current_second > last_recorded_second["value"]:
                last_recorded_second["value"] = current_second
                info["trajectory"].append({
                    "time": current_second,
                    "obj": best,
                    "gap": gap
                })
    return cb


def make_callback_new(solve_id: str, metrics_list):
    """返回一个回调函数，该回调会把当前 solve 的时间和 gap 存到 metrics_list."""
    info = {"solve_id": solve_id,
            "hit": False,  # 是否已经记录过达到1%gap
            "time_at_1pct": None,
            "gap_at_hit": None,
            "time_perf_counter": None,

            # 关于1000s时的数据
            "hit_1000": False,  # 是否已记录过 1000 秒状态
            "obj_at_1000": None,
            "gap_at_1000": None,

            # 每一秒的数据
            "trajectory": [],  # 每一秒记录 {"time": ..., "obj": ..., "gap": ...}
            }

    metrics_list.append(info)
    # 使用字典来模拟引用，确保在闭包中可以修改
    last_recorded_second = {"value": -1}
    start = time.perf_counter()

    def cb(model, where):
        if where == GRB.Callback.MIP:
            # 这是一个优化的回调函数，它避免在每次调用时都做大量计算

            elapsed = model.cbGet(GRB.Callback.RUNTIME)
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)

            # 计算 Gap
            if abs(best) > 1e-6:
                gap = abs(best - bound) / abs(best)
            else:
                gap = float("inf")

            # --- 目标 A: 记录达到 1% Gap 的时间 ---
            if gap <= 0.01 and not info["hit"]:
                info["hit"] = True
                now = time.perf_counter()
                info["time_perf_counter"] = now
                info["time_at_1pct"] = now - start
                info["gap_at_hit"] = gap

            # --- 目标 B: 记录 1000 秒时的状态 ---
            if elapsed >= 1000 and not info["hit_1000"]:
                info["hit_1000"] = True
                info["obj_at_1000"] = best
                info["gap_at_1000"] = gap

            # --- 目标 C: 记录每秒的求解轨迹 ---
            current_second = int(elapsed)
            if current_second > last_recorded_second["value"]:
                last_recorded_second["value"] = current_second
                info["trajectory"].append({
                    "time": current_second,
                    "obj": best,
                    "gap": gap
                })

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


def get_solving_cache(cache:dict,cache_file:str,directory: str, num_problems: int,Threads:int,time_limit=3600):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # log folder
    log_folder = directory.replace('./instance/', './log/')
    log_folder = log_folder + f"_threads_{Threads}"
    os.makedirs(log_folder,exist_ok=True)

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
            model_orig.setParam('LogFile', os.path.join(log_folder,f'{lp_file}.log') )
            model_orig.setParam("TimeLimit", time_limit)

            # 记录是否已经输出过信息
            cb = make_callback(lp_file, all_metrics)
            t0 = time.perf_counter()
            model_orig.optimize(cb)
            t1 = time.perf_counter()

            # 最优解
            Vars = model_orig.getVars()
            solution = {var.VarName: var.X for var in Vars}

            # 指标
            obj_sense = model_orig.ModelSense
            status_orig = model_orig.Status
            obj_orig = model_orig.ObjVal
            time_orig = t1 - t0
            var_num = model_orig.getAttr("NumVars")
            constr_num = model_orig.getAttr("NumConstrs")
            # 写入缓存
            cache[lp_path] = {
                'time_limit':time_limit,
                'obj_sense':   obj_sense,
                'status_orig': status_orig,
                'obj_orig':    obj_orig,
                'time_orig':   time_orig,
                'var_num':var_num,
                'constr_num':constr_num,
                'solution':solution,
                'hit_1000':all_metrics[-1]["hit_1000"],
                "obj_at_1000":all_metrics[-1]["obj_at_1000"],
                "gap_at_1000":all_metrics[-1]["gap_at_1000"],
                'gap_at_hit_1pct': all_metrics[-1]['gap_at_hit'],
                'hit_1pct_gap': all_metrics[-1]['hit'],
                'time_at_1pct': all_metrics[-1]['time_at_1pct'],
                'every_second':all_metrics[-1]['trajectory']
            }

            # 保存log
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

def load_optimal_cache(cache_file, lp_files_dir, solve_num, Threads=0,time_limit=3600):
    # 加载缓存（如果存在）
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    get_solving_cache(cache,cache_file,lp_files_dir, solve_num,Threads,time_limit)
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

def get_problem_parameters(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            problem_parameters = json.load(f)
        print(f"成功从 {json_file_path} 加载参数。")
    except FileNotFoundError:
        print(f"错误: 找不到参数文件 {json_file_path}。")
        exit(1)  # 找不到文件
    except json.JSONDecodeError:
        print(f"错误: {json_file_path} 文件格式不正确。")
        exit(1)  # JSON格式错误
    return problem_parameters

def get_post_fix(param):
    param_values_str = [str(v) for v in param.values()]
    post_fix = "_".join(param_values_str)
    return post_fix


def aggregate_constr(model_agg,agg_num=None,sample=None):
    # 对于sample出的约束，要分为大于等于、小于等于和等于
    # sample是约束
    # todo
    conss = model_agg.getConstrs()

    if sample == None:
        sample = random.sample(conss, min(agg_num, len(conss)))
    if agg_num == None:
        agg_num = 50
        print("using default agg num")

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


def solve_single_lp(lp_path: str, lp_file: str, log_folder: str, threads_per_solve: int, time_limit: int):
    """
    在单独的进程中求解一个 .lp 文件。
    返回: (lp_path, result_dict) 或 (lp_path, None)
    """
    # 打印进程ID，方便调试
    print(f"------------ [PID: {os.getpid()}] 开始处理 {lp_file}-------------")
    try:
        # **关键**: 针对本次求解，创建一个本地列表来接收指标
        # all_metrics_local: List[Dict[str, Any]] = []
        all_metrics_local = []

        # 读入模型
        # Gurobi 的许可证需要在每个进程中都有效
        model_orig = gp.read(lp_path)

        # 参数设置
        model_orig.setParam("Threads", threads_per_solve)
        model_orig.setParam('LogFile', os.path.join(log_folder, f'{lp_file}.log'))
        model_orig.setParam("TimeLimit", time_limit)

        # 传入本地列表
        cb = make_callback_new(lp_file, all_metrics_local)
        t0 = time.perf_counter()
        model_orig.optimize(cb)
        t1 = time.perf_counter()

        # 指标提取
        obj_sense = model_orig.ModelSense
        status_orig = model_orig.Status
        obj_orig = model_orig.ObjVal if status_orig == GRB.OPTIMAL or status_orig == GRB.TIME_LIMIT else None
        time_orig = t1 - t0
        var_num = model_orig.getAttr("NumVars")
        constr_num = model_orig.getAttr("NumConstrs")

        # 从本地列表中提取回调指标
        if not all_metrics_local:
            # 如果回调没有运行或没有添加任何内容（例如，模型读取失败）
            metrics_data = {
                'hit_1000': None, "obj_at_1000": None, "gap_at_1000": None,
                'gap_at_hit_1pct': None, 'hit_1pct_gap': None,
                'time_at_1pct': None, 'every_second': []
            }
        else:
            last_metric = all_metrics_local[-1]
            # 这里的键名必须与主函数中期望写入缓存的键名完全一致
            metrics_data = {
                'hit_1000': last_metric.get("hit_1000"),
                "obj_at_1000": last_metric.get("obj_at_1000"),
                "gap_at_1000": last_metric.get("gap_at_1000"),
                'gap_at_hit_1pct': last_metric.get('gap_at_hit'),
                'hit_1pct_gap': last_metric.get('hit'),
                'time_at_1pct': last_metric.get('time_at_1pct'),
                'every_second': last_metric.get('trajectory', [])
            }

        # 准备要返回的缓存数据
        result_dict = {
            'time_limit': time_limit,
            'obj_sense': obj_sense,
            'status_orig': status_orig,
            'obj_orig': obj_orig,
            'time_orig': time_orig,
            'var_num': var_num,
            'constr_num': constr_num,
            **metrics_data  # 合并指标
        }

        print(f"------------ [PID: {os.getpid()}] 完成 {lp_file}-------------")
        return (lp_path, result_dict)

    except Exception as e:
        # 捕获任何求解或Gurobi错误
        print(f"!!!!!!!! [PID: {os.getpid()}] 求解 {lp_file} 失败: {e}")
        return (lp_path, None)


def get_solving_cache_parallel(
        cache: dict,
        cache_file: str,
        directory: str,
        num_problems: int,
        threads_per_solve: int,
        num_parallel_solves: int,  # 同时运行的进程数
        time_limit: int = 3600
) -> dict:
    # 1. 获取和过滤文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()
    lp_files = lp_files[:num_problems]

    # 2. log folder
    log_folder = directory.replace('./instance/', './log/')
    os.makedirs(log_folder, exist_ok=True)

    # 3. 识别需要求解的任务
    tasks_to_submit = []
    for lp_file in lp_files:
        lp_path = os.path.join(directory, lp_file)
        if lp_path not in cache:
            tasks_to_submit.append((lp_path, lp_file, log_folder, threads_per_solve, time_limit))
        else:
            print(f"Processing {lp_file} cache... 已在缓存中，跳过。")

    if not tasks_to_submit:
        print("所有指定的问题都已在缓存中。")
        return cache

    print(f"总共发现 {len(tasks_to_submit)} 个新问题需要求解。将启动 {num_parallel_solves} 个进程。")

    # 4. 使用 ProcessPoolExecutor 并行执行任务
    cache_updated = False

    # ProcessPoolExecutor 管理进程的创建和销毁
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_solves) as executor:

        future_to_lp_path = {
            # 提交任务， solve_single_lp 及其参数会被序列化并发送给子进程
            executor.submit(solve_single_lp, *task_args): task_args[0]
            for task_args in tasks_to_submit
        }

        # 5. 收集结果
        for future in concurrent.futures.as_completed(future_to_lp_path):
            lp_path = future_to_lp_path[future]
            try:
                result = future.result()

                # result 是 (lp_path, result_dict)
                if result and result[1]:
                    returned_lp_path, result_dict = result
                    # **关键**: 在主进程中更新 cache
                    cache[returned_lp_path] = result_dict
                    cache_updated = True
                    print(f"✅ 结果已缓存: {os.path.basename(returned_lp_path)}")
                else:
                    print(f"❌ 任务失败或无结果，未缓存: {os.path.basename(lp_path)}")

            except Exception as exc:
                print(f'🔴 {os.path.basename(lp_path)} 生成了意料之外的异常: {exc}')

    # 6. **重要**: 统一保存 cache 文件
    if cache_updated:
        print("\n所有任务完成。正在保存更新后的缓存到文件...")
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"🎉 缓存已成功保存到 {cache_file}")
        except Exception as e:
            print(f"!!!!!!!! 严重：保存缓存文件 {cache_file} 失败: {e}")
    else:
        print("\n没有新的结果需要保存。")

    return cache


def load_cache(cache_file,task_name) -> dict:
    # os.makedirs(cache_dir,exist_ok=True)
    # cache_file = os.path.join(cache_dir,f'{task_name}_solve_cache.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 缓存文件 {cache_file} 损坏或为空，将创建新缓存。")
            return {}
    return {}