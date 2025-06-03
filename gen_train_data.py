from gurobipy import *
import numpy as np
import argparse
import pickle
import random
import time
import os
import gurobipy as gp
import utils

def z_score_normalize(lst):
    if not lst:
        return []
    mean = sum(lst) / len(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    std_dev = variance ** 0.5
    if std_dev == 0:  # 处理所有元素相同的情况
        return [0.0] * len(lst)
    return [(x - mean) / std_dev for x in lst]
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


def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type):
    '''
    Function description:
    Solves a problem instance using the Gurobi solver based on the provided inputs.

    Parameter descriptions:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] indicates the number of decision variables involved in the ith constraint.
    - site: site[i][j] indicates which decision variable is involved in the jth position of the ith constraint.
    - value: value[i][j] indicates the coefficient of the jth decision variable in the ith constraint.
    - constraint: constraint[i] indicates the right-hand side value of the ith constraint.
    - constraint_type: constraint_type[i] indicates the type of the ith constraint, where 1 represents <= and 2 represents >=.
    - coefficient: coefficient[i] represents the coefficient of the ith decision variable in the objective function.
    - time_limit: Maximum time allowed for solving.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    - lower_bound: lower_bound[i] represents the lower bound of the range for the ith decision variable.
    - upper_bound: upper_bound[i] represents the upper bound of the range for the ith decision variable.
    - value_type: value_type[i] represents the type of the ith decision variable, 'B' indicates a binary variable, 'I' indicates an integer variable, 'C' indicates a continuous variable.
    '''

    # Get the start time
    begin_time = time.time()
    # Define the optimization model
    model = Model("Gurobi")
    # Define n decision variables x[]
    x = []
    for i in range(n):
        if(value_type[i] == 'B'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
        elif(value_type[i] == 'C'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
        else:
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
    # Set the objective function and optimization goal (maximize/minimize)
    coeff = 0
    for i in range(n):
        coeff += x[i] * coefficient[i]
    if(obj_type == 'maximize'):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    # Add m constraints
    for i in range(m):
        constr = 0
        for j in range(k[i]):
            #print(i, j, k[i])
            constr += x[site[i][j]] * value[i][j]
        if(constraint_type[i] == 1):
            model.addConstr(constr <= constraint[i])
        elif(constraint_type[i] == 2):
            model.addConstr(constr >= constraint[i])
        else:
            model.addConstr(constr == constraint[i])
    # Set the maximum solving time
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    # Optimize the solution
    model.optimize()
    ans = []
    for i in range(n):
        if(value_type[i] == 'C'):
            ans.append(x[i].X)
        else:
            ans.append(int(x[i].X))
    return ans


def constr_sample(lp_path,total_rounds,k):
    model = gp.read(lp_path)
    sample_list = []
    constr_num = model.getAttr("NumConstrs")

    items = list(range(constr_num))
    random.shuffle(items)

    # 计算完整子集数量与剩余元素
    num_full = constr_num // k
    rem = constr_num % k
    blocks = []
    # 切分完整子集
    for i in range(num_full):
        start = i * k
        end = start + k
        blocks.append(set(items[start:end]))  # List slicing citeturn0search1

    # 处理剩余元素，若有剩余则补缺为完整子集
    if rem > 0:
        last_block = set(items[num_full * k:])  # 剩余元素
        # 从全集中随机补充 (k - rem) 个元素
        to_add = set(random.sample(range(constr_num), k - rem))  # random.sample 无放回抽样 citeturn0search3
        last_block.update(to_add)
        blocks.append(last_block)

    # 若初步子集超过 total_rounds，截取前 total_rounds 个
    if len(blocks) >= total_rounds:
        return blocks[:total_rounds]

    # 随机补缺至 total_rounds
    for _ in range(total_rounds - len(blocks)):
        blocks.append(set(random.sample(range(constr_num), k)))  # 随机抽样补缺 citeturn0search3

    return blocks



def gen_constr_label(lp_path,cache,sample_round,k):
    # 约束sampler

    sample_list = constr_sample(lp_path,sample_round,k)

    time_reduce_list = []

    # read cache
    if lp_path in cache:
        print("------------read cache-------------")
        entry = cache[lp_path]
        obj_sense = entry['obj_sense']
        status_orig = entry['status_orig']
        obj_orig = entry['obj_orig']
        time_orig = entry['time_orig']
    else:
        raise Exception("no cache")

    constr_num = None
    # 对于每一个约束子集
    subset_and_timereduce = [] # 元素是列表，列表第一项存放约束index，第二项存放原时间，第三项存放agg后时间

    for sample in sample_list:
        # 聚合

        model = gp.read(lp_path)
        t0 = time.perf_counter()
        ## todo,暂时设置
        # model.setParam("TimeLimit", 2)
        model.Params.OutputFlag = 0

        # 不能放在聚合后
        if constr_num == None:
            constr_num = model.getAttr("NumConstrs")

        conss = model.getConstrs()
        sample_conss = [constr for idx,constr in enumerate(conss) if idx in sample]
        utils.aggregate_constr(model,k,sample_conss)

        # 求解
        model.optimize()
        t1 = time.perf_counter()

        ## 计算分数
        agg_time = t1-t0
        time_reduce = (time_orig - agg_time)/time_orig
        time_reduce_list.append(time_reduce)

        # 存放原始结果
        subset_and_timereduce.append([sample,time_orig,agg_time])

    # 计算约束的分数
    score = {idx:{"total_score":0,"cnt":0} for idx in range(constr_num)}
    for idx,sample in enumerate(sample_list):
        for constr_idx in sample:
            score[constr_idx]["total_score"] += time_reduce_list[idx]
            score[constr_idx]["cnt"] += 1

    score = [[key,val["total_score"]/val["cnt"]] for key,val in score.items()]
    score.sort(key=lambda x:x[1],reverse=True)

    return  score,subset_and_timereduce




def optimize(
    time: int,
    number: int,
):

    task_name = "CA_500_600"
    lp_dir_path = f"./instance/train/{task_name}"
    os.makedirs(f"./parsed/train/{task_name}", exist_ok=True)
    lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
    lp_files.sort()  # 按文件名排序，确保顺序一致

    # cache
    cache_dir = "./cache/train"
    cache = utils.load_optimal_cache(cache_dir, task_name, lp_dir_path, solve_num=len(lp_files), Threads=0)

    # dataset存放目录
    dataset_dir = f"./dataset/{task_name}"
    os.makedirs(dataset_dir,exist_ok=True)

    #
    BG_folder = "/BG"
    constr_score_folder = "/constr_score_multiplier_1"
    solve_info_folder = "/solve_info"

    # 依次读取并每个 .lp 文件
    for lp_file in lp_files:

        # # 如果已经有了constr_score，就跳过
        if os.path.exists(dataset_dir + constr_score_folder +f"/{lp_file.rsplit('.',1)[0]}_constr_score" + '.pickle'):
            print("continue")
            continue

        # 读取问题
        lp_path = os.path.join(lp_dir_path, lp_file)
        model = gp.read(lp_path)


        # obj sense
        obj_type = model.ModelSense
        if obj_type == GRB.MINIMIZE:
            obj_type = 'minimize'
        elif obj_type == GRB.MAXIMIZE:
            obj_type = 'maximize'
        else:
            raise Exception("unknown obj sense")

        ## 描述
        # n represents the number of decision variables
        # m represents the number of constraints
        # k[i] represents the number of decision variables in the ith constraint
        # site[i][j] represents the decision variable in the jth position of the ith constraint
        # value[i][j] represents the coefficient of the jth decision variable in the ith constraint
        # constraint[i] represents the right-hand side value of the ith constraint
        # constraint_type[i] represents the type of the ith constraint, where 1 is <=, 2 is >=
        # coefficient[i] represents the coefficient of the ith decision variable in the objective function
        # lower_bound[i] represents the lower bound of the range for the ith decision variable.
        # upper_bound[i] represents the upper bound of the range for the ith decision variable.
        # value_type[i] represents the type of the ith decision variable, 'B' for binary variable, 'I' for integer variable, 'C' for continuous variable.


        # 获取变量信息
        vars = model.getVars()
        n = len(vars)
        coefficient = [v.obj for v in vars]
        lower_bound = [v.lb for v in vars]
        upper_bound = [v.ub for v in vars]
        value_type = [{'B': 'B', 'I': 'I', 'C': 'C'}.get(v.vtype, 'C') for v in vars]

        # 获取约束信息
        constrs = model.getConstrs()
        m = len(constrs)
        sense_map = {'<': 1, '>': 2, '=': 3}
        k, site, value, constraint, constraint_type = [], [], [], [], []

        constr_degree = []
        variable_degree = [0 for i in range(n)]
        for c in constrs:
            row = model.getRow(c)
            vars_in_row = [row.getVar(idx) for idx in range(row.size())]
            coeffs = [row.getCoeff(idx) for idx in range(row.size())]

            k.append(len(vars_in_row))
            site.append([v.index for v in vars_in_row]) # 变量下标
            value.append([float(co) for co in coeffs])
            constraint.append(c.RHS)
            constraint_type.append(sense_map[c.Sense])
            constr_degree.append(row.size()) # 度

            for idx in range(row.size()):
                var = row.getVar(idx)
                variable_degree[var.index] += 1

        norm_variable_degree = z_score_normalize(variable_degree)
        norm_constr_degree = z_score_normalize(constr_degree)



        # 这个函数是用来得到最优解
        # optimal_solution = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time, obj_type, lower_bound, upper_bound, value_type)

        # Bipartite graph encoding
        variable_features = []
        constraint_features = []
        edge_indices = [[], []] 
        edge_features = []
        # variable_features:目标系数，变量标识[0,1]（区别于约束），变量类型(0,1代表连续或整数)，随机特征，加一个变量的表示吧
        # 再加：在所有变量中的平均系数、度、最大系数、最小系数、每个变量出现顺序的二进制编码

        # constraint_features：约束的平均变量系数、约束的度、右端项、sense


        #print(value_type)
        # 归一化
        norm_coeff = z_score_normalize(coefficient)
        ## 变量原始特征
        for i in range(n):
            now_variable_features = []
            # now_variable_features.append(coefficient[i])
            now_variable_features.append(norm_coeff[i]) # 归一化的系数
            now_variable_features.append(0) #
            now_variable_features.append(1) # [0,1]代表是变量

            # 变量类型
            if(value_type[i] == 'C'):
                now_variable_features.append(0)
            else:
                now_variable_features.append(1)
            # 随机特征
            now_variable_features.append(random.random())

            # 变量的度
            now_variable_features.append(norm_variable_degree[i])

            # 还差：在约束中的平均系数，最大系数，最小系数
            variable_features.append(now_variable_features)

        # 约束原始特征
        for i in range(m):
            now_constraint_features = []
            now_constraint_features.append(constraint[i]) # 右端项

            # 约束类型
            if(constraint_type[i] == 1):
                now_constraint_features.append(1)
                now_constraint_features.append(0)
                now_constraint_features.append(0)
            if(constraint_type[i] == 2):
                now_constraint_features.append(0)
                now_constraint_features.append(1)
                now_constraint_features.append(0)
            if(constraint_type[i] == 3):
                now_constraint_features.append(0)
                now_constraint_features.append(0)
                now_constraint_features.append(1)
            # 随机特征
            now_constraint_features.append(random.random())

            # 度
            now_constraint_features.append(norm_constr_degree[i])

            # pos_emb
            pos_emb = decimal_to_binary_list(n,i)
            now_constraint_features.extend(pos_emb)
            constraint_features.append(now_constraint_features)
        
        for i in range(m):
            for j in range(k[i]):
                edge_indices[0].append(i)
                edge_indices[1].append(site[i][j])
                edge_features.append([value[i][j]])


        ## 约束标签采集
        sample_round = 40 # 采样此时
        k = 50 # 约束子集大小
        constr_score, subset_and_timereduce = gen_constr_label(lp_path, cache,sample_round,k)

        # 保存
        os.makedirs(dataset_dir + BG_folder,exist_ok=True)
        os.makedirs(dataset_dir + constr_score_folder, exist_ok=True)
        os.makedirs(dataset_dir + solve_info_folder, exist_ok=True)
        with open(dataset_dir + BG_folder + f"/{lp_file.rsplit('.',1)[0]}_BG" + '.pickle', 'wb') as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features], f)
        with open(dataset_dir + constr_score_folder + f"/{lp_file.rsplit('.',1)[0]}_constr_score" + '.pickle', 'wb') as f:
                pickle.dump([constr_score], f)
        with open(dataset_dir + solve_info_folder + f"/{lp_file.rsplit('.',1)[0]}_solve_info" + '.pickle', 'wb') as f:
                pickle.dump([subset_and_timereduce], f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type = int, default = 10, help = 'Running wall-clock time.')
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args() # 用不上
    #print(vars(args))
    optimize(**vars(args))

