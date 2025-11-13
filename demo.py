import os
import pickle
import utils
import json
import gurobipy as gp
import time
from gurobipy import GRB
import pickle

def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

filename = "./dataset/CA_500_600/solve_info/CA_0_solve_info.pickle"
cache = read_pickle_file(filename)
print(cache)






# 读问题
# task_name = "CA_700_1100_0.65"
# lp_dir_path = f"./instance/test/{task_name}"
#
# lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
# lp_files.sort()  # 按文件名排序，确保顺序一致
#
# vars = []
# constrs = []
# res = {}
# for lp_file in lp_files:
#
#
#     # 读取问题
#     lp_path = os.path.join(lp_dir_path, lp_file)
#
#     model = gp.read(lp_path)
#     t0 = time.perf_counter()
#     model.optimize()
#     t1 = time.perf_counter()
#     res[lp_file] = {"time":t1-t0}
#     print(res)

    # vars.append(len(model.getVars()))
    # constrs.append(len(model.getConstrs()))
# print("vars:", sum(vars)/len(vars))
# print("constrs:",sum(constrs)/len(constrs))




