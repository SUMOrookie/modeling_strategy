import os
import gurobipy as gp
import utils
import random
import math
from gurobipy import GRB,LinExpr
import repair_func


problem = "CA"
data_dir = "CA_500_600"
directory = f"./instance/test/{data_dir}_new_constr"
# directory = f"./instance/test/{data_dir}"

# 获取目录下所有的 .lp 文件
lp_files = [f for f in os.listdir(directory) if f.endswith('.lp')]
lp_files.sort()  # 按文件名排序，确保顺序一致

# m = gp.read("./instance/test/CA_500_600_new_constr/CA_14_new_constr_75.lp")
# m.optimize()
random.seed(1)
for lp_file in lp_files:
    # 得到lp路径
    lp_path = os.path.join(directory, lp_file)
    print(f"Processing {lp_file}")


    # 生成新约束
    # model = gp.read(lp_path)
    # # num_new_constr = len(model.getConstrs())//10
    # rhs_value = 1
    # utils.generate_and_save_feasible_model(lp_path, directory,
    #                                  initial_rhs=1,
    #                                  initial_frac=0.5,
    #                                  seed=42)
    #
    #
    # continue

    # 聚合
    model_agg = gp.read(lp_path)
    agg_num = 50
    utils.aggregate_constr(model_agg,agg_num)

    ## 求解
    model_agg.setParam("TimeLimit", 5)
    # model_agg.setParam('SolutionLimit', 1)
    model_agg.setParam('OutputFlag', 0)
    model_agg.optimize()

    ### 修复

    # 获得解值
    Vars = model_agg.getVars()
    value_dict = {var.VarName: var.X for var in Vars}

    # 读入新模型，用于修复
    repair_model = gp.read(lp_path)
    # for varname,x in value_dict.items():
    #     print(varname,"\t",x)
    repair_func.heuristic_repair_light_MILP(repair_model,value_dict,lp_path)

    fix_cnt = 0
    unfix_cnt = 0
    for val in value_dict.values():
        if val == None:
            unfix_cnt+=1
        else:
            fix_cnt+=1
    print(f"fix:{fix_cnt},unfix:{unfix_cnt}")


    """
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
            if lhs_value > rhs:
                valid = 0
                # 违反，全部释放？
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

        post_model = gp.read(lp_path)

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
            print(post_model.status)
            post_model.computeIIS()
            # 2. 收集所有被标记为 IIS 的约束
            infeasible_constrs = []
            for constr in post_model.getConstrs():
                if constr.IISConstr:  # 属性为 True 表示该约束属于 IIS
                    row = post_model.getRow(constr)
                    name_list = [row.getVar(idx).VarName for idx in range(row.size())]
                    for var_name in name_list:
                        value_dict[var_name] = None
                    # infeasible_constrs.append(constr.ConstrName)
                    # row = post_model.getRow(constr)
                    # print(row,end=" ")
                    # print(constr.Sense, constr.RHS)
                    # name_list = [row.getVar(idx).VarName for idx in range(row.size())]
                    # for var_name in name_list:
                    #     print(var_name,":",value_dict[var_name],end=" ")
                    # print("---")
            # raise Exception("不可行")

"""













