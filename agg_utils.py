import time
from gurobipy import GRB

def get_agg_constr(model_agg,output,idx_train,agg_num):
    constr_score = output[idx_train].detach().numpy().tolist()
    constr_idx_score =  [[idx,item[1]]for idx,item in enumerate(constr_score)]
    constr_idx_score.sort(key=lambda x:x[1], reverse=True)

    agg_constr_idx = [constr_idx_score[i][0] for i in range(agg_num)]

    # 聚合求解
    conss = model_agg.getConstrs()
    sample = [conss[constr_idx] for constr_idx in agg_constr_idx]


def solve_agg_instance(post_solve_method,model_agg,fix_var,Threads,seed,args):
    # get reduced cost
    reduced_costs = None
    if post_solve_method == "neighborhood_improved":
        model_agg_relax = model_agg.copy()
        model_agg_relax = model_agg_relax.relax()
        model_agg_relax.optimize()
        reduced_costs = [v.RC for v in model_agg_relax.getVars()]
    if fix_var:
        # 备份，用于固定变量后的求解
        model_agg_2 = model_agg.copy()
        agg_model_solve_time = args.agg_model_solve_time
        # 聚合求解
        model_agg_2.setParam("Threads",Threads)
        model_agg_2.setParam("Seed", seed+1)
        model_agg_2.setParam("TimeLimit", agg_model_solve_time)
        model_agg_2.optimize()

        # 获得聚合问题的解（满足整数性）
        Vars = model_agg_2.getVars()
        vaule_dict = {var.VarName: var.X for var in Vars}

        ##  松弛
        for v in model_agg.getVars():
            try:
                v.VType = GRB.CONTINUOUS
            except Exception:
                # 如果直接赋值失败，忽略（继续）
                pass
        model_agg.update()

        # 求解松弛聚合问题
        model_agg.setParam("Threads",Threads)
        model_agg.setParam("Seed", seed)
        if agg_model_solve_time == -1:
            model_agg.optimize()
        else:
            model_agg.setParam("TimeLimit", agg_model_solve_time)
            model_agg.optimize()

        reduced_costs = [v.RC for v in model_agg.getVars()]
        cnt = 0
        if sum([1 if r < 0 else 0 for r in reduced_costs]) != 0:
            threshold = sum([r if r < 0 else 0 for r in reduced_costs]) / sum([1 if r < 0 else 0 for r in reduced_costs])
            # threshold = calculate_threshold(reduced_costs,0.25)
            fixed_vars = [v.VarName for v in model_agg.getVars() if v.RC <= threshold]
            for varname in fixed_vars:
                if vaule_dict[varname] != 0:
                    cnt+=1
                vaule_dict[varname] = 0
            print(f"额外固定了：{cnt}")
        else:
            print(f"固定0个变量,因为reduced_cost均为0,无指导意义")

    else:
        model_agg.setParam("Threads", Threads)
        agg_model_solve_time = args.agg_model_solve_time
        if agg_model_solve_time == -1:
            model_agg.optimize()
        else:
            model_agg.setParam("TimeLimit", agg_model_solve_time)
            model_agg.optimize()

        # 获得变量值
        Vars = model_agg.getVars()
        vaule_dict = {var.VarName: var.X for var in Vars} 
    return vaule_dict,reduced_costs