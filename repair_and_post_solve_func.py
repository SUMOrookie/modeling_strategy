import random
import time
from gurobipy import GRB
import gurobipy as gp
import utils
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

    # alpha = 0.5 todo 需要调
    alpha = 0.5
    if sense == GRB.MAXIMIZE:
        # 最大化问题
        for var_name, n_j in violation_count.items():
            n_j = violation_count.get(var_name, 0)
            c_j = varname_objc_map.get(var_name, 0.0)
            norm_nj = n_j / max_violation
            norm_cj = c_j / max_obj_coeff

            # score计算
            # norm_nj越大代表变量导致更多的约束不可行，最大化问题中norm_cj越大表示目标系数越大，那么对于最大化问题，应该尽可能保留这些变量，因此降低score
            score = alpha * norm_nj - (1-alpha) * norm_cj
            scores[var_name] = score
    elif sense == GRB.MINIMIZE:
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
        # todo，适配不同类型的不等式，以及不同数值的右端项。
        if N > 1:
            fix_num = int(N - 1)
            for i in range(fix_num):
                var_name = var_vaule_one[i]
                vaule_dict[var_name] = 0
    return vaule_dict


def heuristic_repair_subproblem(repair_model,value_dict):
    print("------------repair-------------")

    # 复制模型，用来构造子问题
    sub = repair_model.copy()

    # 根据 value_dict 删除取0的变量
    zeros = [v for v in sub.getVars() if value_dict.get(v.VarName, 0) == 0]
    for v in zeros:
        sub.remove(v)
    sub.update()

    # 清理空约束：若某约束不再包含任何变量，则删除它
    for c in sub.getConstrs():
        row = sub.getRow(c)
        if row.size() == 0:
            sub.remove(c)
    sub.update()

    # 求解子模型
    sub.optimize()

    # 6. 将子模型解写回 value_dict
    if sub.Status == GRB.OPTIMAL or sub.Status == GRB.FEASIBLE:
        for v in sub.getVars():
            # 保留变量在子模型中可能被改为0
            value_dict[v.VarName] = int(round(v.X))

    return value_dict



def heuristic_repair_light_MILP(repair_model,value_dict,lp_path):
    conss = repair_model.getConstrs()

    for constr in conss:
        row = repair_model.getRow(constr)
        var_coeffs_in_constr = {row.getVar(idx).VarName:row.getCoeff(idx) for idx in range(row.size())}
        rhs = constr.RHS
        sense = constr.Sense
        flag = 0
        lhs_value = 0.0
        # 累加左侧值
        for varname,coeff in var_coeffs_in_constr.items():
            if value_dict[varname] == None:
                flag = 1
            else:
                lhs_value += value_dict[varname] * coeff
        # print("lhs_value：",lhs_value)
        # if lhs_value == 6:
        #     print(var_coeffs_in_constr)
        #     print(sense)
        # 根据约束类型判断是否违反
        if sense == "<":
            # var_order = [[varname, value_dict[varname] * var_coeffs_in_constr[varname]] for varname in
            #              var_coeffs_in_constr.keys() if value_dict[varname] != None]
            # var_order.sort(key=lambda x: x[1], reverse=False)  # 越靠后越大
            if lhs_value > rhs:
                valid = 0
                # 并非全部释放
                # item = var_order.pop()
                # value_dict[item[0]] = None
                # lhs_value -= item[1]
                # 违反，全部释放
                for varname in var_coeffs_in_constr.keys():
                    if value_dict[varname] != None:
                        value_dict[varname] = None

        elif sense == ">":
            if lhs_value + flag < rhs:
                valid = 0
                # 违反，全部释放
                for varname in var_coeffs_in_constr.keys():
                    if value_dict[varname] != None:
                        value_dict[varname] = None

        else:
            raise Exception("unknown sense")

    # 验证解是否可行
    while True:

        #
        post_model = gp.read(lp_path)
        post_model.setParam('OutputFlag', 0)
        for var_name, value in value_dict.items():
            var = post_model.getVarByName(var_name)
            if var is None:
               raise Exception(f"变量 {var_name} 不存在于模型中")

            # 添加等式约束：var == value
            # print(var_name, ":", value)
            if value == None:
                pass
            else:
                post_model.addConstr(var == int(value), f"fix_{var_name}")
        post_model.update()
        post_model.optimize()

        if post_model.status == GRB.OPTIMAL:
            print("解是可行的（满足所有约束和变量上下界）")
            break
        else:
            print("status:",post_model.status)
            post_model.computeIIS()
            # 2. 收集所有被标记为 IIS 的约束
            infeasible_constrs = []
            for constr in post_model.getConstrs():
                all_none = True
                if constr.IISConstr:  # 属性为 True 表示该约束属于 IIS
                    row = post_model.getRow(constr)
                    name_list = [row.getVar(idx).VarName for idx in range(row.size())]
                    for var_name in name_list:
                        if value_dict[var_name] != None:
                            value_dict[var_name] = None
                            all_none = False
                    if not all_none:
                       break
                    # infeasible_constrs.append(constr.ConstrName)
                    # row = post_model.getRow(constr)
                    # print(row,end=" ")
                    # print(constr.Sense, constr.RHS)
                    # name_list = [row.getVar(idx).VarName for idx in range(row.size())]
                    # for var_name in name_list:
                    #     print(var_name,":",value_dict[var_name],end=" ")
                    # print("---")
            # raise Exception("不可行")
    return value_dict


def PostSolve(repair_model,neighborhood,vaule_dict,lp_file,start_time,time_limit):
    ## 修复后，作为原模型初始解求解，也就是再接入求解器
    print("------------PostSolve-------------")
    # 赋初始值
    k0, k1, Delta = neighborhood["k0"], neighborhood["k1"], neighborhood["Delta"]
    k0_cnt = 0
    k1_cnt = 0
    delta_var_list = []

    repair_Vars = repair_model.getVars()
    for idx in range(len(repair_Vars)):
        varname = repair_Vars[idx].VarName
        if vaule_dict[varname] != None:
            # 给初始值
            # repair_model.getVarByName(varname).Start = vaule_dict[varname]

            # 直接固定
            # var = repair_model.getVarByName(varname)
            # var.LB = vaule_dict[varname]
            # var.UB = vaule_dict[varname]

            # 加邻域的固定
            if abs(vaule_dict[varname]) == 0.0 and k0_cnt < k0:
                var = repair_model.getVarByName(varname)
                var_delta = repair_model.addVar(vtype=GRB.BINARY,name=f"k0_{k0_cnt}")
                repair_model.addConstr(var<=var_delta,name=f"region_k0_{k0_cnt}")
                k0_cnt+=1
                delta_var_list.append(var_delta)
            elif vaule_dict[varname] == 1.0 and k1_cnt < k1:
                var = repair_model.getVarByName(varname)
                var_delta = repair_model.addVar(vtype=GRB.BINARY, name=f"k1_{k1_cnt}")
                repair_model.addConstr((1-var) <= var_delta, name=f"region_k1_{k1_cnt}")
                k1_cnt += 1
                delta_var_list.append(var_delta)
            else:
                # 进入此处的可能：变量取值不为0-1；k0与k1已经达到上限
                break
    print(f"k0:{k0_cnt},\t,k1:{k1_cnt}")

    # 半径约束
    repair_model.addConstr(
        gp.quicksum(d for d in delta_var_list) <= Delta,
        name="radius"
    )

    repair_model.update()

    # 求解,计算指标
    # repair_model.setParam("TimeLimit", 2)
    # cb = utils.make_callback(lp_file, utils.all_metrics)
    # repair_model.setParam("Seed", 1234)
    rest_time = time_limit - (time.perf_counter() - start_time)
    repair_model.setParam("TimeLimit", rest_time)
    # repair_model.optimize(cb)
    repair_model.optimize()
