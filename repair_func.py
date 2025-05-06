from gurobipy import GRB


def compute_scores(model, varname_objc_map,value_dict, violation_count, c_mean,sense):
    """
    计算每个取值为1变量的评分：
      score_j = -C_j + (n_j - 1) * c_mean

    参数:
      model           : Gurobi 模型
      value_dict      : {var_name: value} 的映射
      violation_count : {var_name: n_j} 的映射，n_j 为在不可行约束中出现的次数
      c_mean          : 全局目标系数均值

    返回:
      scores : {var_name: score_j} 的映射
    """

    scores = {}
    max_violation = max(violation_count.values() if violation_count else 1)
    obj_coeff = varname_objc_map.values()
    obj_coeff = [abs(x) for x in obj_coeff]
    max_obj_coeff = max(obj_coeff)
    if max_obj_coeff == 0:
        max_obj_coeff = 1

    # alpha = 0.5
    alpha = 0
    if sense == GRB.MAXIMIZE:
        # 最大化问题
        for var_name, n_j in violation_count.items():
            n_j = violation_count.get(var_name, 0)
            c_j = varname_objc_map.get(var_name, 0.0)
            norm_nj = n_j / max_violation
            norm_cj = c_j / max_obj_coeff

            # score计算
            score = alpha * norm_nj - (1-alpha) * norm_cj
            scores[var_name] = score
    elif sense == GRB.MAXIMIZE:
        raise Exception("还没写呢")
    else:
        raise Exception("unknown sense")
    return scores


def heuristic_repair_with_score(model, value_dict):
    """
    使用基于 score_j 的启发式修复：
      1. 统计 violation_count
      2. 计算 c_mean
      3. 调用 compute_scores 得分并排序
      4. 依次将分数最高的变量设为0，直至所有约束满足
    """
    # 1. 统计每个取1的变量在不可行约束中的出现次数
    violation_count = {}
    violation_constr_info = {}
    lhs_values = {}
    RHS_values = {}
    for constr_idx, constr in enumerate(model.getConstrs()):
        # 计算左端式子的值
        lhs = 0.0  # 约束表达式的值
        row = model.getRow(constr)  # 得到LinExpr
        var_in_constr = [row.getVar(idx).VarName for idx in range(row.size())]  # 获得约束里的变量名
        var_vaule_one = [] # 存下这个约束中，取值为1的变量。
        for var_idx , var_name in enumerate(var_in_constr):
            if var_name in value_dict and value_dict[var_name] == 1:
                lhs += value_dict[var_name] * row.getCoeff(var_idx)
                var_vaule_one.append(var_name)

        if lhs > constr.RHS + 1e-6: # 在CA问题中，RHS=1
            lhs_values[constr_idx] = lhs
            RHS_values[constr_idx] = constr.RHS
            # violation_count要把var_vaule_one中的变量的值全部加1
            for varname in var_vaule_one:
                violation_count[varname] = violation_count.get(varname, 0) + 1
                violation_constr_info.setdefault(varname, []).append(constr_idx)



    if not violation_count:
        return value_dict  # 已全部满足

    # 2. 计算全局目标系数均值 c_mean
    obj = model.getObjective()

    all_coeffs = [obj.getCoeff(idx) for idx in range(obj.size())]
    c_mean = sum(all_coeffs) / len(all_coeffs)

    varname_objc_map = {obj.getVar(idx).VarName:obj.getCoeff(idx) for idx in range(obj.size())}

    # 3. 计算评分并排序
    sense = model.ModelSense
    scores = compute_scores(model, varname_objc_map,value_dict, violation_count, c_mean,sense)
    # 分数越高优先置0
    sorted_vars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    constrs = model.getConstrs()

    # 4. 依次置0并检查
    for var_name, score in sorted_vars:
        value_dict[var_name] = 0

        # 找到这个变量违反的约束
        for idx in violation_constr_info.get(var_name, []):
            constr =constrs[idx]  # 通过索引直接获取约束 :contentReference[oaicite:1]{index=1}
            row = model.getRow(constr)  # 得到 LinExpr :contentReference[oaicite:2]{index=2}

            var_in_constr = [[row.getVar(idx).VarName,row.getCoeff(idx)] for idx in range(row.size())]
            for var_idx, alist in enumerate(var_in_constr):
                if var_name == alist[0]:
                    lhs_values[idx] -= alist[1]
                    break
        if all(lhs_values[idx] <= RHS_values[idx] for idx in lhs_values.keys()):
            break

    return value_dict


def heuristic_repair(repair_model,vaule_dict):
    print("------------repair-------------")

    conss = repair_model.getConstrs()

    # 启发式修复
    for constr in conss:
        N = 0.0  # 约束表达式的值
        row = repair_model.getRow(constr)  # 得到LinExpr
        var_in_constr = [row.getVar(idx).VarName for idx in range(row.size())]  # 获得约束里的变量名
        var_vaule_one = []  # 保存取值为1的变量名
        for var_name in var_in_constr:
            if var_name in vaule_dict and vaule_dict[var_name] == 1:
                N += vaule_dict[var_name]
                var_vaule_one.append(var_name)

        # 当约束表达式的值大于1（右端项是1），仅保留一个变量取1，其余取1的变量变为0
        if N > 1:
            fix_num = int(N - 1)
            for i in range(fix_num):
                var_name = var_vaule_one[i]
                vaule_dict[var_name] = 0
    return vaule_dict