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
import pyscipopt
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE
from typing import List,Dict,Tuple,Set


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


def make_callback_scip(solve_id, metrics_list):
    """
    SCIP 版本的回调工厂函数。

    它会创建指标字典，将其附加到 metrics_list，
    然后返回一个配置好的 ScipMetricsCallback *实例*。
    """
    info = {
        "solve_id": solve_id,
        "hit": False,
        "time_at_1pct": None,
        "gap_at_hit": None,
        "time_perf_counter": None,

        "hit_1000": False,
        "obj_at_1000": None,
        "gap_at_1000": None,

        "trajectory": [],
    }

    metrics_list.append(info)

    # 获取 perf_counter 的起始时间
    start = time.perf_counter()

    # 创建并返回 Event Handler 实例
    return ScipMetricsCallback(info_dict=info, start_time_perf_counter=start)



def summary(repair_model,t0,obj_sense,bks,gurobi_time,every_second,gurobi_obj,results,
            lp_file,status_orig,cons_num_orig,cons_num_agg,bks_time):
    t1 = time.perf_counter()


    status_agg = repair_model.Status
    obj_agg = repair_model.ObjVal
    total_time_agg = t1 - t0

    ## gap、时间约简计算
    if obj_sense == GRB.MINIMIZE:
        print("最小化问题")
        primal_gap =  100 * (obj_agg - bks) / abs(bks) if bks != 0 else float("inf")
    else:
        print("最大化问题")
        primal_gap =  100 * (bks - obj_agg) / abs(bks) if bks != 0 else float("inf") 
    time_reduce = (gurobi_time - total_time_agg) / gurobi_time if gurobi_time > 0 else 0

    # gain计算
    # obj_orig_same_time = every_second[round(total_time_agg)+1]['obj']
    by_time = {item["time"]: item for item in every_second}
    at_time_info = by_time.get(int(total_time_agg),None)
    if at_time_info is not None:
        obj_orig_same_time = at_time_info.get('obj',None)

    ## 计算差值
    # if obj_sense == GRB.MINIMIZE:
    #     print("最小化问题")
    #     if at_time_info is not None:
    #         gap_abs_orig = obj_orig_same_time - obj_orig
    #         gap_abs_agg = obj_agg - obj_orig
    #
    # else:
    #     print("最大化问题")
    #     if at_time_info is not None:
    #         gap_abs_orig =  obj_orig - obj_orig_same_time
    #         gap_abs_agg = obj_orig - obj_agg


    
    print(f"原obj:{gurobi_obj},\t 聚合后obj:{obj_agg}")
    print(f"原时间:{gurobi_time},\t 聚合后时间:{total_time_agg}")
    print(f"primal_gap:{primal_gap}")
    print(f"time_reduce:{time_reduce}")
    if at_time_info is not None:
        gap_at_same_time = at_time_info.get('gap',None)
        print(f"gap_at_same_time:{gap_at_same_time}")
    # 保存
    if at_time_info is not None:
        results.append({
            "filename": lp_file,
            "status_orig": status_orig,
            "status_agg": status_agg,
            "obj_orig": bks,
            "obj_gurobi": gurobi_obj,
            "obj_agg": obj_agg,
            "primal_gap(%)": primal_gap,
            # "obj_orig_same_time":obj_orig_same_time,
            # 'gap_at_same_time(full_gap)':gap_at_same_time,
            # 'gap_abs_orig':gap_abs_orig,
            # 'gap_abs_agg':gap_abs_agg,
            "original_cons_num": cons_num_orig,
            "after_cons_num": cons_num_agg,
            "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
            "bks_time":bks_time,
            "time_gurobi": gurobi_time,
            "time_agg": total_time_agg,
            "time_reduce": time_reduce,
            "original_cons": cons_num_orig,
            "after_agg_cons": cons_num_agg,
        })
    else:
        results.append({
            "filename": lp_file,
            "status_orig": status_orig,
            "status_agg": status_agg,
            "obj_orig": bks,
            "obj_gurobi": gurobi_obj,
            "obj_agg": obj_agg,
            "primal_gap(%)": primal_gap,
            "original_cons_num": cons_num_orig,
            "after_cons_num": cons_num_agg,
            "cons_reduce_ratio": (cons_num_orig - cons_num_agg) / cons_num_orig,
            "bks_time": bks_time,
            "time_gurobi": gurobi_time,
            "time_agg": total_time_agg,
            "time_reduce": time_reduce,
            "original_cons": cons_num_orig,
            "after_agg_cons": cons_num_agg,
        })

class ScipMetricsCallback(Eventhdlr):
    """
    SCIP 事件处理器 (Event Handler)，用于复刻 Gurobi 回调的指标记录功能。

    它将跟踪：
    1. 达到 1% MipGap 的时间。
    2. 求解 1000 秒时的状态。
    3. 每一秒的 (time, obj, gap) 轨迹。
    """

    def __init__(self, info_dict, start_time_perf_counter):
        """
        初始化 Event Handler。

        参数:
        info_dict (dict): 从外部传入的、用于存储指标的字典。
        start_time_perf_counter (float): time.perf_counter() 的起始时间。
        """
        super().__init__()
        self.info = info_dict
        self.start_time = start_time_perf_counter
        self.last_recorded_second = -1

    def eventinit(self):
        """
        SCIP 回调：初始化事件。
        我们在这里 "catch" (订阅) 我们关心的事件。
        """
        # SCIP_EVENTTYPE.BESTSOLFOUND: 每当找到新的*整数*解（Primal Bound 改善）时触发。
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        # SCIP_EVENTTYPE.LPSOLVED: 每次 LP 松弛被求解时触发（Dual Bound 改善）时触发。
        self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexit(self):
        """
        SCIP 回调：退出事件。
        取消订阅事件。
        """
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, event):
        """
        SCIP 回调：执行事件。
        这是 Gurobi `cb` 函数的主体逻辑所在。
        每当 BESTSOLFOUND 或 LPSOLVED 事件之一被触发时，此方法将被调用。
        """

        # 1. 获取数据 (等同于 Gurobi 的 cbGet)

        # GRB.Callback.RUNTIME -> model.getSolvingTime()
        # 这是求解器内部的运行时间
        elapsed = self.model.getSolvingTime()

        # GRB.Callback.MIP_OBJBST -> model.getPrimalbound()
        # 这是目前为止找到的最好的整数解的目标值
        best = self.model.getPrimalbound()

        # GRB.Callback.MIP_OBJBND -> model.getDualbound()
        # 这是全局的双边界
        bound = self.model.getDualbound()

        # 2. 计算 Gap
        gap = float('inf')
        # 必须检查是否已经找到了至少一个解 (getNSols() > 0)
        # 否则 getPrimalbound() 会返回 'inf'
        if self.model.getNSols() > 0:
            if abs(best) > 1e-6:
                gap = abs(best - bound) / abs(best)
            else:
                # 如果 best 接近 0，则仅当 bound 也接近 0 时 gap 才为 0
                gap = 0.0 if abs(bound) < 1e-6 else float('inf')

        # 3. 执行 Gurobi 回调中的逻辑

        # 逻辑 A：记录 1% Gap
        if gap <= 0.01 and not self.info["hit"]:
            self.info["hit"] = True
            now = time.perf_counter()
            self.info["time_perf_counter"] = now
            self.info["time_at_1pct"] = now - self.start_time
            self.info["gap_at_hit"] = gap

        # 逻辑 B：记录 1000 秒时的状态
        if elapsed >= 1000 and not self.info["hit_1000"]:
            self.info["hit_1000"] = True
            # 记录1000s时的状态，如果还没找到解，记为 nan
            self.info["obj_at_1000"] = best if self.model.getNSols() > 0 else float('nan')
            self.info["gap_at_1000"] = gap

        # 逻辑 C：每整秒记录一次
        current_second = int(elapsed)
        if current_second > self.last_recorded_second:
            self.last_recorded_second = current_second
            self.info["trajectory"].append({
                "time": current_second,
                "obj": best if self.model.getNSols() > 0 else float('nan'),
                "gap": gap
            })

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
            slack = [[cons.index,cons.Slack] for cons in model_orig.getConstrs()]
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
                "slack": slack,
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

all_metrics_scip = []
def get_solving_cache_scip(cache:dict,cache_file:str,directory: str, num_problems: int,Threads:int,time_limit=3600):
    # 获取目录下所有的 .lp 文件
    lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # 限制读取的文件数量
    lp_files = lp_files[:num_problems]

    # log folder
    log_folder = directory.replace('./instance/', './log/')
    log_folder = log_folder + f"_threads_{Threads}_scip"
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
            # model_orig = gp.read(lp_path)
            model_scip = pyscipopt.Model()
            model_scip.readProblem(lp_path)

            # 时间从读入开始算,求解
            # model_scip.setParam('limits/nthreads', Threads)
            model_scip.setLogfile(os.path.join(log_folder, f'{lp_file}.log'))
            # model_scip.setParam('misc/logfile', os.path.join(log_folder, f'{lp_file}.log'))
            model_scip.setParam('limits/time', time_limit)


            # 3. 求解
            # 注册回调函数 (Gurobi: optimize(cb) -> SCIP: Event Handler)
            # SCIP 使用 Event Handler 来实现回调功能。
            # 您需要创建一个继承自 Eventhdlr 类的对象，并重载其事件处理方法。
            # 这里我们假设 make_callback 已经更新为返回一个适用于 SCIP 的 Event Handler 实例。

            # 假设 make_callback 现在返回一个 SCIP Event Handler 实例
            scip_callback_handler = make_callback_scip(lp_file, all_metrics_scip)
            model_scip.includeEventhdlr(scip_callback_handler,
                                        "PyMetricsCallback",  # 回调的任意名称
                                        "Python metrics logger")  # 任意描述

            t0 = time.perf_counter()
            model_scip.optimize()
            t1 = time.perf_counter()

            # 4. 提取结果
            time_scip = t1 - t0
            status_scip = model_scip.getStatus()
            var_num = model_scip.getNVars()
            constr_num = model_scip.getNConss()

            # 检查是否找到解
            if model_scip.getNSols() > 0:
                # 提取最优解
                sol = model_scip.getBestSol()
                solution = {var.name: model_scip.getSolVal(sol, var) for var in model_scip.getVars()}

                # 目标值
                obj_scip = model_scip.getObjVal()
            else:
                solution = {}
                # 如果没有找到解，目标值可能是无限大或-无限大
                obj_scip = float('nan')
                print(f"SCIP Status: {status_scip}. No solution found.")

            # 目标方向 (SCIP: minimize=1, maximize=-1)
            # Gurobi: minimize=1, maximize=-1
            obj_sense_scip = model_scip.getObjectiveSense()

            # 5. SCIP 状态转换 (简化)
            # 常用 SCIP 状态:
            #   - SCIP_STATUS.OPTIMAL: 找到最优解
            #   - SCIP_STATUS.INFEASIBLE: 不可行
            #   - SCIP_STATUS.TIMELIMIT: 达到时间限制
            #   - SCIP_STATUS.UNKNOWN: 求解过程终止，但状态未知（例如，内存限制）
            # 您可能需要一个映射函数将 SCIP_STATUS 转换为 Gurobi 类似的整数或自定义字符串。
            # 为了保持与原代码结构一致，我们直接使用 SCIP_STATUS 的 name
            status_scip_name = status_scip

            # 6. 写入缓存
            cache[lp_path] = {
                'time_limit': time_limit,
                'obj_sense': obj_sense_scip,
                'status_orig': status_scip_name,  # 使用 SCIP 的状态名
                'obj_orig': obj_scip,
                'time_orig': time_scip,
                'var_num': var_num,
                'constr_num': constr_num,
                'solution': solution,
                'hit_1000': all_metrics_scip[-1]["hit_1000"],
                "obj_at_1000": all_metrics_scip[-1]["obj_at_1000"],
                "gap_at_1000": all_metrics_scip[-1]["gap_at_1000"],
                'gap_at_hit_1pct': all_metrics_scip[-1]['gap_at_hit'],
                'hit_1pct_gap': all_metrics_scip[-1]['hit'],
                'time_at_1pct': all_metrics_scip[-1]['time_at_1pct'],
                'every_second': all_metrics_scip[-1]['trajectory']
            }

            # 7. 保存log
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

def get_cache_file_path(dataset_name,data_dir):
    cache_dir = f"./cache/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, f'{data_dir}.json')
    return cache_file_path



def load_optimal_cache_scip(cache_file, lp_files_dir, solve_num, Threads=0,time_limit=3600):
    # 加载缓存（如果存在）
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    get_solving_cache_scip(cache,cache_file,lp_files_dir, solve_num,Threads,time_limit)
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


def aggregate_constr_scip(model_agg: Model, agg_num=None, sample=None):
    # 对于sample出的约束，要分为大于等于、小于等于和等于
    # sample是约束
    # todo
    conss = model_agg.getConss()
    if agg_num == None:
        agg_num = 50
        print("using default agg num")
    if sample == None:
        sample = random.sample(conss, min(agg_num, len(conss)))

    # 乘子
    # primes = utils.gen_primes(agg_num)
    # u_list = [math.log(p) for p in primes]
    var_map = {var.name: var for var in model_agg.getVars()}

    u_list = [1 for i in range(len(sample))]
    # 计算聚合约束
    agg_coeffs_leq = {}
    agg_rhs_leq = 0.0
    agg_coeffs_geq = {}
    agg_rhs_geq = 0.0
    agg_coeffs_eq = {}
    agg_rhs_eq = 0.0
    inf = model_agg.infinity()
    for idx, cons in enumerate(sample):
        u = u_list[idx]
        # SCIP stores constraints as ranged rows: lhs <= a^T x <= rhs
        lhs = model_agg.getLhs(cons)
        rhs = model_agg.getRhs(cons)

        # Try to fetch linear coefficients and variables.
        # Note: getValsLinear works only for linear constraints (see docs).
        try:
            vars_in_cons = model_agg.getConsVars(cons)
            vals = model_agg.getValsLinear(cons).values()
        except Exception as e:
            # if constraint is not a plain linear constraint (e.g. was specialized by presolve),
            # skip it (or you could try to transform/skip-presolve beforehand).
            print(f"Skipping non-linear/special constraint {cons} due to: {e}")
            continue

        # Decide sense:
        # equality if lhs and rhs are (almost) equal
        if lhs == rhs:
            sense = "=="
        else:
            # lhs == -infty -> only rhs side present => <= rhs
            if lhs <= -0.5 * inf:
                sense = "<="
            # rhs == +infty -> only lhs side present => >= lhs
            elif rhs >= 0.5 * inf:
                sense = ">="
            else:
                # ranged constraint (finite lhs and rhs, but lhs != rhs).
                # We'll treat it as both a >= lhs and a <= rhs (i.e., two-sided).
                sense = "range"

        # accumulate coefficients
        for var, coef in zip(vars_in_cons, vals):
            name = var.name
            if sense == "<=":
                agg_coeffs_leq[name] = agg_coeffs_leq.get(name, 0.0) + u * coef
            elif sense == ">=":
                agg_coeffs_geq[name] = agg_coeffs_geq.get(name, 0.0) + u * coef
            elif sense == "==":
                agg_coeffs_eq[name] = agg_coeffs_eq.get(name, 0.0) + u * coef
            elif sense == "range":
                agg_coeffs_leq[name] = agg_coeffs_leq.get(name, 0.0) + u * coef
                agg_coeffs_geq[name] = agg_coeffs_geq.get(name, 0.0) + u * coef

        # accumulate rhs / lhs appropriately
        if sense == "<=":
            agg_rhs_leq += u * rhs
        elif sense == ">=":
            agg_rhs_geq += u * lhs
        elif sense == "==":
            agg_rhs_eq += u * rhs  # lhs==rhs so rhs is fine
        elif sense == "range":
            agg_rhs_leq += u * rhs
            agg_rhs_geq += u * lhs
        else:
            raise Exception("unknown constraint sense")

        # remove the original constraint from the model
        model_agg.delCons(cons)

    # build and add aggregated constraints if there are any terms
    if agg_coeffs_leq:
        expr = pyscipopt.quicksum(
            coef * var_map[name] for name, coef in agg_coeffs_leq.items()
        )
        model_agg.addCons(expr <= agg_rhs_leq, name="agg_constraint_leq")

    if agg_coeffs_geq:
        expr = pyscipopt.quicksum(
            coef * var_map[name] for name, coef in agg_coeffs_geq.items()
        )
        model_agg.addCons(expr >= agg_rhs_geq, name="agg_constraint_geq")

    if agg_coeffs_eq:
        expr = pyscipopt.quicksum(
            coef * var_map[name] for name, coef in agg_coeffs_eq.items()
        )
        model_agg.addCons(expr == agg_rhs_eq, name="agg_constraint_eq")

    #     constr_expr = model_agg.getRow(cons)
    #     sense = cons.Sense
    #     for j in range(constr_expr.size()):
    #         var = constr_expr.getVar(j)
    #         coef = constr_expr.getCoeff(j)
    #         if sense == "<":
    #             agg_coeffs_leq[var.VarName] = agg_coeffs_leq.get(var.VarName, 0.0) + u * coef
    #         elif sense == ">":
    #             agg_coeffs_geq[var.VarName] = agg_coeffs_geq.get(var.VarName, 0.0) + u * coef
    #         elif sense == "=":
    #             agg_coeffs_eq[var.VarName] = agg_coeffs_eq.get(var.VarName, 0.0) + u * coef
    #         else:
    #             raise Exception("unknown constr sense")
    #     if sense == "<":
    #         agg_rhs_leq += u * cons.RHS
    #     elif sense == ">":
    #         agg_rhs_geq += u * cons.RHS
    #     elif sense == "=":
    #         agg_rhs_eq += u * cons.RHS
    #     else:
    #         raise Exception("unknown constr sense")
    #     model_agg.remove(cons)  # 删除约束
    # model_agg.update()
    #
    # # 构造聚合约束
    # expr_leq = 0
    # expr_geq = 0
    # expr_eq = 0
    # for var_name, coef in agg_coeffs_leq.items():
    #     var = model_agg.getVarByName(var_name)
    #     expr_leq += coef * var
    # for var_name, coef in agg_coeffs_geq.items():
    #     var = model_agg.getVarByName(var_name)
    #     expr_geq += coef * var
    # for var_name, coef in agg_coeffs_eq.items():
    #     var = model_agg.getVarByName(var_name)
    #     expr_eq += coef * var
    #
    # model_agg.addConstr(expr_leq <= agg_rhs_leq, name="agg_constraint_leq")
    # model_agg.addConstr(expr_geq >= agg_rhs_geq, name="agg_constraint_geq")
    # model_agg.addConstr(expr_eq == agg_rhs_eq, name="agg_constraint_eq")
    # model_agg.update()


def lp_relax_and_solve(model: gp.Model, time_limit: float = 1e3, silent: bool = True) -> gp.Model:
    """
    输入：一个 Gurobi Model（可能是 MIP）
    输出：一个已经 relax() 并 optimize() 完成的 LP 模型（copy），
          若失败则抛出异常或返回 None
    """
    # 复制模型，避免破坏原始
    lp = model.copy()

    # 松弛
    for v in lp.getVars():
        try:
            v.VType = GRB.CONTINUOUS
        except Exception:
            # 如果直接赋值失败，忽略（继续）
            pass
    lp.update()
    lp.optimize()
    status = lp.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise gp.GurobiError(f"LP relaxation solve status {status}")
    return lp

def extract_constraint_info(lp: gp.Model) -> Tuple[List[gp.Constr], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    从求解后的 LP 模型中提取：
      - constraints：约束对象列表
      - duals_map：约束名称 -> 对偶值 (Pi)
      - rows_map：约束名称 -> { var_name: coeff, ... }（仅保留非零系数）
    """
    # 获取 pi (对偶)
    constrs = lp.getConstrs()
    duals = lp.getAttr("Pi")
    duals_map = {} #
    rows_map: Dict[str, Dict[str, float]] = {}

    for c, pi in zip(constrs, duals):
        name = c.ConstrName if c.ConstrName is not None else f"c_{c.getAttr('Index')}"
        duals_map[name] = float(pi)

        # 获取约束的行向量（非零项）
        # 使用 getRow：返回 LinExpr 或 Row，可迭代地查看变量和系数
        try:
            row = lp.getRow(c)
            row_entries = {}
            for j in range(row.size()):
                v = row.getVar(j)
                coeff = row.getCoeff(j)
                row_entries[v.VarName] = float(coeff)
        except gp.GurobiError:
            # 如果 getRow 不可用，退化为空
            row_entries = {}
        rows_map[name] = row_entries

    return list(duals_map.keys()), duals_map, rows_map

def normalize_duals(duals_map: Dict[str, float],
                    rows_map: Dict[str, Dict[str, float]],
                    method: str = "l1") -> Dict[str, float]:
    """
    归一化对偶分数，返回 score_map: name -> score（绝对值/归一化）
    method: "l1" (|pi| / sum|a_ij|), "l2" (|pi| / sqrt(sum a_ij^2)), "rhs" (|pi| * |b_i|), "none"
    需要注意：rows_map 仅包含系数，不包含 RHS；若想用 RHS，需额外提取（这里假设 RHS unavailable）
    """
    score_map = {}
    for name, pi in duals_map.items():
        abs_pi = abs(pi)
        row = rows_map.get(name, {})
        if method == "l1":
            denom = sum(abs(v) for v in row.values()) or 1.0
            score = abs_pi / denom
        elif method == "l2":
            denom = math.sqrt(sum((v ** 2) for v in row.values())) or 1.0
            score = abs_pi / denom
        elif method == "rhs":
            # 如果你想用 RHS，需要先把 RHS 做到 rows_map 或单独提取。暂时退化为 abs_pi
            score = abs_pi
        else:
            score = abs_pi
        score_map[name] = float(score)
    return score_map

def select_strong_constraints(score_map: Dict[str, float], keep_fraction: float) -> Tuple[Set[str], Set[str]]:
    """
    按 score 排序，保留 top keep_fraction 的约束（返回保留集和弱约束集）
    """
    n = len(score_map)
    k = max(1, int(math.ceil(n * keep_fraction)))
    # sort by descending score
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    strong = set(name for name, _ in sorted_items[:k])
    weak = set(score_map.keys()) - strong
    return strong, weak

# 相似度计算工具
def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def cosine_similarity_from_dicts(dict_a: Dict[str, float], dict_b: Dict[str, float]) -> float:
    # dot / (||a|| * ||b||)
    # dot: sum over intersection of coeff_a * coeff_b
    common = set(dict_a.keys()) & set(dict_b.keys())
    if not common:
        return 0.0
    dot = sum(dict_a[k] * dict_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in dict_a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in dict_b.values())) or 1.0
    return dot / (norm_a * norm_b)


def compute_similarity_matrix(weak_names: List[str],
                              rows_map: Dict[str, Dict[str, float]],
                              metric: str ) -> Dict[Tuple[str, str], float]:
    """
    计算弱约束集两两相似度，返回字典 {(name1, name2): sim}
    注意：这是 O(m^2) 的操作，m = len(weak_names)。对大问题请谨慎。
    metric: "jaccard", "cosine", or "both"
    """
    sim_dict: Dict[Tuple[str, str], float] = {}
    m = len(weak_names)
    print(f"[info] computing pairwise similarity for {m} weak constraints (O(m^2) pairs = {m*(m-1)//2})")

    # Precompute sets for jaccard and dicts for cosine
    sets = {name: set(rows_map[name].keys()) for name in weak_names}
    dicts = {name: rows_map[name] for name in weak_names}

    # iterate pairs
    for i in range(m):
        name_i = weak_names[i]
        for j in range(i + 1, m):
            name_j = weak_names[j]
            if metric == "jaccard":
                s = jaccard_similarity(sets[name_i], sets[name_j])
            elif metric == "cosine":
                s = cosine_similarity_from_dicts(dicts[name_i], dicts[name_j])
            elif metric == "both":
                s_j = jaccard_similarity(sets[name_i], sets[name_j])
                s_c = cosine_similarity_from_dicts(dicts[name_i], dicts[name_j])
                # 合并两种度量供后续选择；这里简单取平均（也可以返回 tuple）
                s = 0.5 * (s_j + s_c)
            else:
                raise ValueError("Unknown metric")
            sim_dict[(name_i, name_j)] = float(s)
            # 可选：对称保存（不必要但方便）
            # sim_dict[(name_j, name_i)] = float(s)
    return sim_dict

def greedy_pair_aggregation(model_agg,sim_dict):
    """
    sim_dict: dict, key=(ci, cj) with ci< cj, val = similarity
    目标：每次选最高 similarity 的 pair（ci,cj），只要这两个都未被聚合，执行聚合
    """
    # 将 pair 根据 similarity 排序 (从大到小)
    pair_list = sorted(sim_dict.items(), key=lambda kv: kv[1], reverse=True)

    aggregated = set()      # 保存已经被聚合过的约束
    done_pairs = []         # 保存真正执行的 pair

    def read_constr_data(cons_obj):
        constr_expr = model_agg.getRow(cons_obj)
        coeffs = {}
        for j in range(constr_expr.size()):
            var = constr_expr.getVar(j)
            coef = constr_expr.getCoeff(j)
            coeffs[var.VarName] = coeffs.get(var.VarName, 0.0) + coef
        rhs = cons_obj.RHS
        return coeffs, rhs

    for (ci, cj), sim in pair_list:
        # 如果任意一个已经被聚合过，跳过
        if ci in aggregated or cj in aggregated:
            continue
        if sim < 0.1:
            continue
        # 执行聚合动作
        ## todo:这里默认是小于等于约束，并且乘子为1
        # 先读取两条约束的数据（在删除之前读取）
        cons1,cons2 = model_agg.getConstrByName(ci),model_agg.getConstrByName(cj)
        coeffs1, rhs1 = read_constr_data(cons1)
        coeffs2, rhs2 = read_constr_data(cons2)

        # 合并系数和 rhs
        agg_coeffs = {}
        for vn, c in coeffs1.items():
            agg_coeffs[vn] = agg_coeffs.get(vn, 0.0) + c
        for vn, c in coeffs2.items():
            agg_coeffs[vn] = agg_coeffs.get(vn, 0.0) + c
        agg_rhs = rhs1 + rhs2

        # 删除原约束（删除后这些 cons 对象变为 removed，但我们已缓存所需数据）
        model_agg.remove(cons1)
        model_agg.remove(cons2)
        model_agg.update()

        # 构造表达式并加入聚合约束
        expr = 0
        for var_name, coef in agg_coeffs.items():
            var = model_agg.getVarByName(var_name)
            expr += coef * var

        name = f"agg_{ci}_{cj}"
        model_agg.addConstr(expr <= agg_rhs, name=name)

        done_pairs.append((ci, cj, sim))

        # 标记两个约束已经聚合过
        aggregated.add(ci)
        aggregated.add(cj)

    return done_pairs, aggregated


def aggregate_constr_duals(model_agg,KEEP_FRACTION,NORM_METHOD,SIMILARITY_METRIC):
    # -------------------------
    OUTPUT_DIR = "./relax_results"
    LP_SOLVE_TIME_LIMIT = 10  # 对 LP relaxation 的求解时限（秒） ,2.0
    VERBOSE = True
    # -------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Step 1: LP relaxation & solve
    lp = lp_relax_and_solve(model_agg, time_limit=LP_SOLVE_TIME_LIMIT, silent=not VERBOSE)

    # Step 2: extract duals and row info
    constraint_names, duals_map, rows_map = extract_constraint_info(lp)

    # Step 3: normalize duals -> scores
    scores = normalize_duals(duals_map, rows_map, method=NORM_METHOD)

    # Step 4: select strong/weak
    strong_set, weak_set = select_strong_constraints(scores, keep_fraction=KEEP_FRACTION)
    weak_list = sorted(list(weak_set))

    if VERBOSE:
        print(f"[info] total constraints: {len(constraint_names)}")
        print(f"[info] strong (kept) constraints: {len(strong_set)}")
        print(f"[info] weak (candidates to aggregate): {len(weak_set)}")

    # Step 5: compute similarity for weak constraints
    if len(weak_list) > 1:
        sim_dict = compute_similarity_matrix(weak_list, rows_map, metric=SIMILARITY_METRIC)
    else:
        sim_dict = {}

    # step 6: 聚合
    pairs, aggregated = greedy_pair_aggregation(model_agg,sim_dict)
    print(f"pairs:{len(pairs)}")


def aggregate_constr_two_two(model_agg, agg_num=None, sample=None):
    """
    两两约束聚合：
    - 从 model_agg 中采样 agg_num 个约束（若 sample 为 None）
    - 按采样顺序每两个一组进行聚合（若为奇数，最后一个不聚合）
    - 每一对内部分别按 <=, >=, = 三类累加并生成对应的聚合约束
    """
    conss = model_agg.getConstrs()

    if sample is None:
        if agg_num is None:
            agg_num = 50
            print("using default agg num")
        sample = random.sample(list(conss), min(agg_num, len(conss)))
    else:
        # 若用户传入 sample，但同时传了 agg_num，遵循 sample 的长度；否则若传了 agg_num 且 sample 为 None，上面会处理
        if agg_num is None:
            agg_num = len(sample)

    # 聚合乘子
    u_list = [1.0] * len(sample)
    weight_map = {cons: u for cons,u in zip(sample,u_list)}

    # 按 sense 分类：注意使用原约束对象作为 key，value 是 (cons, weight)
    geq_cons = []  # '>'
    leq_cons = []  # '<'
    eq_cons = []   # '='

    for cons in sample:
        sense = cons.Sense
        if sense == ">":
            geq_cons.append((cons, weight_map[cons]))
        elif sense == "<":
            leq_cons.append((cons, weight_map[cons]))
        elif sense == "=":
            eq_cons.append((cons, weight_map[cons]))
        else:
            raise Exception(f"unknown constr sense: {sense}")

    def read_constr_data(cons_obj, weight):
        constr_expr = model_agg.getRow(cons_obj)
        coeffs = {}
        for j in range(constr_expr.size()):
            var = constr_expr.getVar(j)
            coef = constr_expr.getCoeff(j)
            coeffs[var.VarName] = coeffs.get(var.VarName, 0.0) + weight * coef
        rhs = weight * cons_obj.RHS
        return coeffs, rhs

    pair_counter = 0

    # 处理顺序：先 >= ，再 <= ，再 =
    for group, sense_tag in ((geq_cons, "geq"), (leq_cons, "leq"), (eq_cons, "eq")):
        # 两两聚合
        idx = 0
        while idx + 1 < len(group):
            cons1, w1 = group[idx]
            cons2, w2 = group[idx + 1]

            # 先读取两条约束的数据（在删除之前读取）
            coeffs1, rhs1 = read_constr_data(cons1, w1)
            coeffs2, rhs2 = read_constr_data(cons2, w2)

            # 合并系数和 rhs
            agg_coeffs = {}
            for vn, c in coeffs1.items():
                agg_coeffs[vn] = agg_coeffs.get(vn, 0.0) + c
            for vn, c in coeffs2.items():
                agg_coeffs[vn] = agg_coeffs.get(vn, 0.0) + c
            agg_rhs = rhs1 + rhs2

            # 删除原约束（删除后这些 cons 对象变为 removed，但我们已缓存所需数据）
            model_agg.remove(cons1)
            model_agg.remove(cons2)
            model_agg.update()

            # 构造表达式并加入聚合约束
            expr = 0
            for var_name, coef in agg_coeffs.items():
                var = model_agg.getVarByName(var_name)
                expr += coef * var

            name = f"agg_{sense_tag}_{pair_counter}"
            if sense_tag == "geq":
                model_agg.addConstr(expr >= agg_rhs, name=name)
            elif sense_tag == "leq":
                model_agg.addConstr(expr <= agg_rhs, name=name)
            elif sense_tag == "eq":
                model_agg.addConstr(expr == agg_rhs, name=name)

            pair_counter += 1
            idx += 2  # 跳到下一对

        # 如果该类别是奇数个，最后一个不处理（保留原约束，不删除）
        # 即：idx == len(group)-1 时，最后一个未被处理也不会被删除或修改
    model_agg.update()

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
        slack = [[cons.index,cons.Slack] for cons in model_orig.getConstrs()]
        Vars = model_orig.getVars()
        solution = {var.VarName: var.X for var in Vars}
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
            'solution':solution,
            "slack": slack,
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


def get_solving_cache_mp(cache: dict,
                         cache_file: str,
                         directory: str,
                         num_problems: int,
                         Threads: int,
                         num_parallel_solves: int = 1,
                         time_limit: int = 3600) -> dict:
    """
    并行封装：基于原始的 `get_solving_cache`，当 `num_parallel_solves > 1` 时
    使用 `get_solving_cache_parallel` 进行多进程求解；否则保持顺序行为。

    参数:
    - cache, cache_file, directory, num_problems: 与原函数一致
    - Threads: 每个求解进程使用的线程数（传给底层求解器）
    - num_parallel_solves: 并行求解的进程数，默认为 1（即顺序）
    - time_limit: 每个求解的时间上限（秒）

    返回: 更新后的 cache 字典并已写入 `cache_file`（如有新结果）
    """
    # 如果需要并行求解，则调用已实现的并行函数
    if num_parallel_solves and num_parallel_solves > 1:
        return get_solving_cache_parallel(cache, cache_file, directory,
                                          num_problems, Threads,
                                          num_parallel_solves, time_limit)
    # 否则保持原先的顺序实现
    return get_solving_cache(cache, cache_file, directory, num_problems, Threads, time_limit)


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