import os
import utils
import gurobipy as gp
from gurobipy import GRB
import random
import time
from EGAT_models import SpGAT
import torch
import argparse
from torch.autograd import Variable
import repair_func

# 读问题
task_name = "CA_500_600"
lp_dir_path = f"./instance/test/{task_name}"

lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
lp_files.sort()  # 按文件名排序，确保顺序一致

# 读取求解cache
cache_dir = "./cache/test"
Threads = 0
solve_num = min(10,len(lp_files))
cache = utils.load_cache(cache_dir, task_name, lp_dir_path, solve_num, Threads)

# parser
# testing settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=16, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=6, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky_relu.')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cpu'

results = []
agg_num = 50
repair_method = "lightmilp"

for lp_file in lp_files[:solve_num]:


    # 读取问题
    lp_path = os.path.join(lp_dir_path, lp_file)

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


    model_agg = gp.read(lp_path)
    t0 = time.perf_counter()

    cons_num_orig = model_agg.getAttr("NumConstrs")

    # obj sense
    obj_type = model_agg.ModelSense
    if obj_type == GRB.MINIMIZE:
        obj_type = 'minimize'
    elif obj_type == GRB.MAXIMIZE:
        obj_type = 'maximize'
    else:
        raise Exception("unknown obj sense")

    # 获取变量信息
    vars = model_agg.getVars()
    n = len(vars)
    coefficient = [v.obj for v in vars]
    lower_bound = [v.lb for v in vars]
    upper_bound = [v.ub for v in vars]
    value_type = [{'B': 'B', 'I': 'I', 'C': 'C'}.get(v.vtype, 'C') for v in vars]

    # 获取约束信息
    constrs = model_agg.getConstrs()
    m = len(constrs)
    sense_map = {'<': 1, '>': 2, '=': 3}
    k, site, value, constraint, constraint_type = [], [], [], [], []
    for c in constrs:
        row = model_agg.getRow(c)
        vars_in_row = [row.getVar(idx) for idx in range(row.size())]
        coeffs = [row.getCoeff(idx) for idx in range(row.size())]

        k.append(len(vars_in_row))
        site.append([v.index for v in vars_in_row])  # 变量下标
        value.append([float(co) for co in coeffs])
        constraint.append(c.RHS)
        constraint_type.append(sense_map[c.Sense])

    # Bipartite graph encoding
    variable_features = []
    constraint_features = []
    edge_indices = [[], []]
    edge_features = []

    # print(value_type)
    for i in range(n):
        now_variable_features = []
        now_variable_features.append(coefficient[i])
        now_variable_features.append(0)  #
        now_variable_features.append(1)  # [0,1]代表变量
        if (value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        now_variable_features.append(random.random())
        variable_features.append(now_variable_features)

    for i in range(m):
        now_constraint_features = []
        now_constraint_features.append(constraint[i])
        if (constraint_type[i] == 1):
            now_constraint_features.append(1)
            now_constraint_features.append(0)
            now_constraint_features.append(0)
        if (constraint_type[i] == 2):
            now_constraint_features.append(0)
            now_constraint_features.append(1)
            now_constraint_features.append(0)
        if (constraint_type[i] == 3):
            now_constraint_features.append(0)
            now_constraint_features.append(0)
            now_constraint_features.append(1)
        now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)

    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])

    # change
    n = len(variable_features)
    var_size = len(variable_features[0])
    m = len(constraint_features)
    con_size = len(constraint_features[0])

    edge_num = len(edge_indices[0])

    edgeA = []
    edgeB = []
    # edge_features = []
    for i in range(edge_num):
        edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
        edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])

    edgeA = torch.as_tensor(edgeA)
    edgeB = torch.as_tensor(edgeB)
    edge_features = torch.as_tensor(edge_features)

    for i in range(m):
        for j in range(var_size - con_size):
            constraint_features[i].append(0)
    features = variable_features + constraint_features
    features = torch.as_tensor(features).float()


    # focal loss的权重
    num_label = [1, 4]
    # num_label = [1, 1]
    num_label = torch.as_tensor(num_label).to(device)

    # 需要计算loss的部分，也就是约束的范围
    idx_train = torch.as_tensor(range(n, n + m))


    ## 过网络
    # 读取网络
    nn_model = SpGAT(nfeat=features.shape[1],  # Feature dimension
                  nhid=args.hidden,  # Feature dimension of each hidden layer
                  nclass=2,  # Number of classes
                  # nclass=int(data_solution[0].max()) + 1,  # Number of classes
                  dropout=args.dropout,  # Dropout
                  nheads=args.nb_heads,  # Number of heads
                  alpha=args.alpha)  # LeakyReLU alpha coefficient

    nn_model.load_state_dict(torch.load(f"./model/{task_name}/model.pkl"))

    if args.cuda:  # Move to GPU
        nn_model.to(device)

        features = features.to(device)

        edgeA = edgeA.to(device)
        edgeB = edgeB.to(device)
        edge_features = edge_features.to(device)
        idx_train = idx_train.to(device)


    features = Variable(features)
    edgeA = Variable(edgeA)
    edgeB = Variable(edgeB)
    # Define computation graph for automatic differentiation

    output, _ = nn_model(features, edgeA, edgeB,edge_features.detach())

    ## 聚合
    # 得到需要聚合的约束
    constr_score = output[idx_train].detach().numpy().tolist()
    constr_idx_score =  [[idx,item[1]]for idx,item in enumerate(constr_score)]
    constr_idx_score.sort(key=lambda x:x[1], reverse=True)


    agg_constr_idx = [constr_idx_score[i][0] for i in range(agg_num)]
    print("aa")
    # 聚合求解
    conss = model_agg.getConstrs()
    sample = [conss[constr_idx] for constr_idx in agg_constr_idx]
    utils.aggregate_constr(model_agg, agg_num, sample)

    cons_num_agg = model_agg.getAttr("NumConstrs")

    print("------------solving agg model-------------")
    model_agg.setParam("Threads", Threads)
    model_solve_time = 2
    if model_solve_time == -1:
        model_agg.optimize()
    else:
        model_agg.setParam("TimeLimit", model_solve_time)
        model_agg.optimize()
    agg_objval_original = model_agg.ObjVal

    # 获得变量值
    Vars = model_agg.getVars()
    vaule_dict = {var.VarName: var.X for var in Vars}

    # 读入新模型，用于修复
    repair_model = gp.read(lp_path)
    repair_model.setParam("Threads", Threads)

    # todo：gurobi修复，并且求到最优解与否
    # 如果gurobi and postsolve：不设置SolutionLimit
    # 如果gurobi and not postsolve：设置SolutionLimit
    # 其他情况，调用对应的启发式函数
    # 然后判断postsolve

    PostSolve = True
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
            vaule_dict = repair_func.heuristic_repair_light_MILP(repair_model, vaule_dict, lp_path)
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
                if vaule_dict[varname] != None:
                    repair_model.getVarByName(varname).Start = vaule_dict[varname]
                # repair_Vars[idx].VarHintVal  = vaule_list[idx]
                # 修复后的作为start，松弛的用hintval

        # 求解（修复）,计算指标
        repair_model.optimize()

        t1 = time.perf_counter()
        status_agg = repair_model.Status
        obj_agg = repair_model.ObjVal
        time_agg = t1 - t0

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
        })
    # 和原始时间对比
result_dir = f"./result/{task_name}"
results.to_csv(result_dir + f"/surrogate_stats_aggnum_{agg_num}__repair_{repair_method}.csv", index=False)
