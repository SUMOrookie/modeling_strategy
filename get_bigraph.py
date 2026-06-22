# import os
# import parser_utils
import utils
# import gurobipy as gp
from gurobipy import GRB
import random
# import time
# from EGAT_models import SpGAT
import torch
from torch.autograd import Variable
# import pandas as pd
# import numpy as np
# from typing import List, Optional
# import json

def get_input(nn_model,args,device,n,m,features,edgeA,edgeB,edge_features):
    # 约束的idx
    idx_train = torch.as_tensor(range(n, n + m))
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
    return features, edgeA, edgeB,edge_features,idx_train

def get_bigraph(model):

    # obj sense
    obj_type = model.ModelSense
    if obj_type == GRB.MINIMIZE:
        obj_type = 'minimize'
    elif obj_type == GRB.MAXIMIZE:
        obj_type = 'maximize'
    else:
        raise Exception("unknown obj sense")

    # -----------------------------
    # 提取变量（columns）信息：目标系数、上下界、类型等
    # -----------------------------
    vars = model.getVars()
    n = len(vars)
    coefficient = [v.obj for v in vars]
    lower_bound = [v.lb for v in vars]
    upper_bound = [v.ub for v in vars]
    value_type = [{'B': 'B', 'I': 'I', 'C': 'C'}.get(v.vtype, 'C') for v in vars]

    # -----------------------------
    # 提取约束（rows）信息：系数位置、RHS、约束类型、行的度（非零个数）等
    # -----------------------------
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
        site.append([v.index for v in vars_in_row])  # 变量下标
        value.append([float(co) for co in coeffs])
        constraint.append(c.RHS)
        constraint_type.append(sense_map[c.Sense])
        constr_degree.append(row.size())  # 度
        for idx in range(row.size()):
            var = row.getVar(idx)
            variable_degree[var.index] += 1

    norm_variable_degree = utils.z_score_normalize(variable_degree)
    norm_constr_degree = utils.z_score_normalize(constr_degree)

    # -----------------------------
    # 将线性规划问题编码为二部图（变量节点 + 约束节点）
    # - variable_features / constraint_features: 节点特征
    # - edge_indices / edge_features: 边与边特征（系数）
    # -----------------------------
    variable_features = []
    constraint_features = []
    edge_indices = [[], []]
    edge_features = []

    # print(value_type)
    norn_coeff = utils.z_score_normalize(coefficient)
    for i in range(n):
        now_variable_features = []
        now_variable_features.append(norn_coeff[i])
        now_variable_features.append(0)  #
        now_variable_features.append(1)  # [0,1]代表变量
        if (value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        now_variable_features.append(random.random())

        # 度
        now_variable_features.append(norm_variable_degree[i])

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

        # 度
        now_constraint_features.append(norm_constr_degree[i])

        # pos_emb
        # pos_emb = utils.decimal_to_binary_list(m, i)
        # now_constraint_features.extend(pos_emb)

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

    if var_size > con_size:
        for i in range(m):
            for j in range(var_size - con_size):
                constraint_features[i].append(0)
    else:
        for i in range(n):
            for j in range(con_size-var_size):
                variable_features[i].append(0)


    features = variable_features + constraint_features
    features = torch.as_tensor(features).float()
    return features,n,m,edgeA,edgeB,edge_features
