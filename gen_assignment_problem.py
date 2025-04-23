import os
import numpy as np


def generate_assignment_lp_instances(num_instances, sizes, cost_low, cost_high, output_dir):
    """
    生成指派问题的 LP 文件，并保存到指定目录。

    参数：
    - num_instances (int): 每种规模要生成的实例数量。
    - sizes (list of int): 要生成的指派矩阵维度列表，例如 [5, 10, 20]。
    - cost_low (int): 成本矩阵中随机生成整数的最小值（含）。
    - cost_high (int): 成本矩阵中随机生成整数的最大值（含）。
    - output_dir (str): 保存 LP 文件的目标文件夹路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    for size in sizes:
        for i in range(1, num_instances + 1):
            # 随机生成 size x size 的成本矩阵
            cost = np.random.randint(cost_low, cost_high + 1, size=(size, size))
            filename = os.path.join(output_dir, f"assignment_{size}x{size}_{i}.lp")

            with open(filename, 'w') as f:
                # 写入目标函数
                f.write("Minimize\n obj: ")
                terms = []
                for r in range(size):
                    for c in range(size):
                        terms.append(f"{cost[r, c]} x_{r}_{c}")
                f.write(" + ".join(terms) + "\n")

                # 写入约束
                f.write("Subject To\n")
                # 每个任务分配给一个代理（行和为1）
                for r in range(size):
                    row_terms = " + ".join(f"x_{r}_{c}" for c in range(size))
                    f.write(f" c_row_{r}: {row_terms} = 1\n")
                # 每个代理执行一个任务（列和为1）
                for c in range(size):
                    col_terms = " + ".join(f"x_{r}_{c}" for r in range(size))
                    f.write(f" c_col_{c}: {col_terms} = 1\n")

                # 变量声明
                f.write("Binary\n")
                for r in range(size):
                    for c in range(size):
                        f.write(f" x_{r}_{c}\n")
                f.write("End\n")

    print(f"已生成 {num_instances * len(sizes)} 个 LP 实例，保存在: {output_dir}")


# 示例调用
output_dir = './instance/test/assignment_lp'
generate_assignment_lp_instances(
    num_instances=2,
    sizes=[5, 10],
    cost_low=1,
    cost_high=100,
    output_dir=output_dir
)

# 列出生成的 LP 文件
import glob

print("生成的文件列表：", glob.glob(os.path.join(output_dir, '*.lp')))
