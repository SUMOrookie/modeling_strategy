import time
import csv
import gurobipy
from gurobipy import GRB
import argparse
import random
import os
import numpy as np
import torch
from helper import get_a_new2
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

project_dir = '/home/cc/code/modeling_strategy/predictandsearch'
#4 public datasets, IS, WA, CA, IP
# TaskName='CA_750_1100_0.7'
TaskName='IS_1500_6'
TestNum=10
def test_hyperparam(task):
    '''
    set the hyperparams
    k_0, k_1, delta
    '''
    if task=="IP":
        return 400,5,1
    elif task == "IS":
        return 300,300,15
    elif task == "WA":
        return 0,600,5
    elif task == "CA_500_600":
        return 400,0,10
    elif task == "CA_750_1100_0.7":
        return 400,30,30
        # return 1,1,2
    elif task == "IS_1500_6":
        return 400,30,30
    else:
        raise NotImplementedError(f"task {task} not implemented")
k_0,k_1,delta=test_hyperparam(TaskName)

#set log folder
solver='GRB'
test_task = f'{TaskName}_{solver}_Predect&Search'
# use project_dir for all repository-local paths
logs_root = os.path.join(project_dir, 'logs')
task_logs_dir = os.path.join(logs_root, TaskName)
test_logs_dir = os.path.join(task_logs_dir, test_task)
os.makedirs(test_logs_dir, exist_ok=True)
log_folder = test_logs_dir


#load pretrained model
if TaskName=="IP":
    #Add position embedding for IP model, due to the strong symmetry
    from GCN import GNNPolicy_position as GNNPolicy,postion_get
else:
    from GCN import GNNPolicy
model_name=f'{TaskName}.pth'
models_dir = os.path.join(project_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
pathstr = os.path.join(models_dir, model_name)
policy = GNNPolicy().to(DEVICE)
state = torch.load(pathstr, map_location=torch.device('cuda:0'))
policy.load_state_dict(state)


# sample_names = sorted(os.listdir(f'./instance/{TaskName}'))
sample_names = sorted(os.listdir(f'/home/cc/code/modeling_strategy/predictandsearch/instance/test/{TaskName}'))
time_list = []
obj_list = []

result_dir = os.path.join(project_dir, 'result', f'{TaskName}_test')
os.makedirs(result_dir, exist_ok=True)

for ins_num in range(TestNum):
    t1 = time.perf_counter()
    test_ins_name = sample_names[ins_num]
    ins_name_to_read = f'/home/cc/code/modeling_strategy/predictandsearch/instance/test/{TaskName}/{test_ins_name}'

    #get bipartite graph as input
    A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(ins_name_to_read)
    constraint_features = c_nodes.cpu()
    constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
    variable_features = v_nodes
    if TaskName == "IP":
        variable_features = postion_get(variable_features)
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)

    #prediction
    BD = policy(
        constraint_features.to(DEVICE),
        edge_indices.to(DEVICE),
        edge_features.to(DEVICE),
        variable_features.to(DEVICE),
    ).sigmoid().cpu().squeeze()

    #align the variable name betweend the output and the solver
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    scores=[]#get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])


    scores.sort(key=lambda x:x[2],reverse=True)

    scores=[x for x in scores if x[4]=='BINARY']#get binary

    fixer=0
    #fixing variable picked by confidence scores
    count1=0
    for i in range(len(scores)):
        if count1<k_1:
            scores[i][3] = 1
            count1+=1
            fixer += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixer += 1


    print(f'instance: {test_ins_name}, '
          f'fix {k_0} 0s and '
          f'fix {k_1} 1s, delta {delta}. ')


    #read instance
    gurobipy.setParam('LogToConsole', 1)  # hideout
    m = gurobipy.read(ins_name_to_read)
    m.Params.TimeLimit = 1000
    # m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = f'{log_folder}/{test_ins_name}.log'

    # trust region method implemented by adding constraints
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue
        # tmp_var = m1.addVar(f'alp_{tar_var}', 'C')
        tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    m.optimize()

    time_list.append(time.perf_counter()-t1)
    obj_list.append(m.ObjVal)

    with open(os.path.join(result_dir,f"ps_result_{k_0}_{k_1}_{delta}.csv"), 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        # 第一次循环时写入表头
        if ins_num == 0:
            csv_writer.writerow(['耗时（s）', '目标值ObjVal'])
        # 写入当前最新的一条数据（也可写入整个列表，按需选择）
        csv_writer.writerow([time_list[-1], obj_list[-1]])  # 只保存本次新增数据

    print(f"time:{time.perf_counter()-t1}")
    print(f"obj:{m.ObjVal}")
for t,o in zip(time_list,obj_list):
    print(f"{t} \t {o}")







