from gurobipy import Model, GRB, quicksum

def solve_integer_programming():
    # 1. 创建模型
    model = Model("IntegerProgrammingExample")

    # 2. 定义变量（整数变量）
    x1 = model.addVar(vtype=GRB.INTEGER, name="x1")
    x2 = model.addVar(vtype=GRB.INTEGER, name="x2")
    x3 = model.addVar(vtype=GRB.INTEGER, name="x3")
    x4 = model.addVar(vtype=GRB.INTEGER, name="x4")

    # 3. 设置目标函数 max Z = x1 + 5x2 + 10x3 + x4
    model.setObjective(
        x1 + 5 * x2 + 10 * x3 + x4,
        GRB.MAXIMIZE
    )

    # 4. 添加约束条件
    # 约束1: x1 + 2x2 + 2x4 ≤ 5
    model.addConstr(x1 + 2 * x2 + 2 * x4 <= 5, "c1")

    # 约束2: x1 + x3 ≤ 1
    model.addConstr(x1 + x3 <= 1, "c2")

    # 约束3: x1 ≤ 2
    model.addConstr(x1 <= 2, "c3")

    # 约束4: x4 ≤ 2
    model.addConstr(x4 <= 2, "c4")

    # （可选）添加非负约束，题目未说明变量非负，所以可以不加
    # 如果题目隐含变量非负，可加上下面这几行
    # model.addConstr(x1 >= 0, "nonneg_x1")
    # model.addConstr(x2 >= 0, "nonneg_x2")
    # model.addConstr(x3 >= 0, "nonneg_x3")
    # model.addConstr(x4 >= 0, "nonneg_x4")

    # 5. 求解模型
    model.optimize()

    # 6. 输出结果
    if model.status == GRB.OPTIMAL:
        print("最优目标值 Z =", model.objVal)
        print("最优解：")
        for var in model.getVars():
            print(f"{var.varName} = {var.x}")
    else:
        print("模型未找到最优解，状态码：", model.status)


def solve_updated_integer_programming():
    # 1. 创建模型
    model = Model("UpdatedIntegerProgrammingExample")

    # 2. 定义整数变量
    x1 = model.addVar(vtype=GRB.INTEGER, name="x1")
    x2 = model.addVar(vtype=GRB.INTEGER, name="x2")
    x3 = model.addVar(vtype=GRB.INTEGER, name="x3")
    x4 = model.addVar(vtype=GRB.INTEGER, name="x4")

    # 3. 设置目标函数 max Z = x1 + 5x2 + 10x3 + x4
    model.setObjective(
        x1 + 5 * x2 + 10 * x3 + x4,
        GRB.MAXIMIZE
    )

    # 4. 添加约束条件
    # 约束1（替代约束）：2x1 + 2x2 + 2x4 ≤ 9
    model.addConstr(2 * x1 + 2 * x2 + 3 * x4 <= 9, "c1")

    # 约束2：x1 + x3 ≤ 1
    model.addConstr(x1 + x3 <= 1, "c2")

    # （可选）如果题目隐含变量非负，添加以下非负约束
    model.addConstr(x1 >= 0, "nonneg_x1")
    model.addConstr(x2 >= 0, "nonneg_x2")
    model.addConstr(x3 >= 0, "nonneg_x3")
    model.addConstr(x4 >= 0, "nonneg_x4")

    # 5. 求解模型
    model.optimize()

    # 6. 输出结果
    if model.status == GRB.OPTIMAL:
        print("最优目标值 Z =", model.objVal)
        print("最优解：")
        for var in model.getVars():
            print(f"{var.varName} = {var.x}")
    else:
        print("模型未找到最优解，状态码：", model.status)


def solve_new_integer_programming():
    # 1. 创建模型
    model = Model("NewIntegerProgrammingExample")

    # 2. 定义整数变量
    x1 = model.addVar(vtype=GRB.INTEGER, name="x1")
    x2 = model.addVar(vtype=GRB.INTEGER, name="x2")
    x3 = model.addVar(vtype=GRB.INTEGER, name="x3")
    x4 = model.addVar(vtype=GRB.INTEGER, name="x4")

    # 3. 设置目标函数 max Z = x1 + 5x2 + 10x3 + x4
    model.setObjective(
        x1 + 5 * x2 + 10 * x3 + x4,
        GRB.MAXIMIZE
    )

    # 4. 添加约束条件
    # 约束1: x1 + 2x2 + 2x4 ≤ 5
    model.addConstr(x1 + 2 * x2 + 2 * x4 <= 5, "c1")

    # 约束2: x1 + x3 ≤ 1
    model.addConstr(x1 + x3 <= 1, "c2")

    # 约束3（替代约束）: x1 + x4 ≤ 4
    model.addConstr(x1 + x4 <= 4, "c3")

    # （可选）添加变量非负约束（题目未明确说明，通常整数规划默认非负）
    model.addConstr(x1 >= 0, "nonneg_x1")
    model.addConstr(x2 >= 0, "nonneg_x2")
    model.addConstr(x3 >= 0, "nonneg_x3")
    model.addConstr(x4 >= 0, "nonneg_x4")

    # 5. 求解模型
    model.optimize()

    # 6. 输出结果
    if model.status == GRB.OPTIMAL:
        print("最优目标值 Z =", model.objVal)
        print("最优解：")
        for var in model.getVars():
            print(f"{var.varName} = {var.x}")
    else:
        print("模型未找到最优解，状态码：", model.status)

if __name__ == "__main__":
    solve_integer_programming()
    solve_updated_integer_programming()
    solve_new_integer_programming()
