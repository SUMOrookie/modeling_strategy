import os
import pickle
import utils
import json
import gurobipy as gp
import time
from gurobipy import GRB









# 读问题
task_name = "CA_same_with_ps"
lp_dir_path = f"./instance/test/{task_name}"

lp_files = [f for f in os.listdir(lp_dir_path) if f.endswith('.lp')]
lp_files.sort()  # 按文件名排序，确保顺序一致

vars = []
constrs = []
for lp_file in lp_files:


    # 读取问题
    lp_path = os.path.join(lp_dir_path, lp_file)

    model = gp.read(lp_path)
    vars.append(len(model.getVars()))
    constrs.append(len(model.getConstrs()))
print("vars:", sum(vars)/len(vars))
print("constrs:",sum(constrs)/len(constrs))




