


# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 6.5), dpi=150)
# plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# # 坐标轴延伸范围，为了展示完整的直线切分
# x_val = np.linspace(-1, 7, 400)
# y_constraint = (10 - 2 * x_val) / 3

# # 1. 绘制核心约束的完整直线 (贯穿坐标系)
# plt.plot(x_val, y_constraint, color='red', linewidth=2, zorder=2, label='连续边界: 2x1 + 3x2 = 10')

# # 2. 严格填充连续可行域 (只在 x1>=0 且 x2>=0 且 满足约束 的区域填充)
# x_fill = np.linspace(0, 5, 200)
# y_fill = (10 - 2 * x_fill) / 3
# plt.fill_between(x_fill, 0, y_fill, color='red', alpha=0.15, zorder=1, label='连续可行域')

# # 加粗 X 轴和 Y 轴，强调第一象限的非负约束边界
# plt.axvline(x=0, color='black', linewidth=1.5, zorder=2)
# plt.axhline(y=0, color='black', linewidth=1.5, zorder=2)

# # 3. 绘制离散整数点
# for i in range(7):
#     for j in range(5):
#         if 2*i + 3*j <= 10:
#             if i == 0 and j == 3:
#                 # 整数最优解
#                 plt.scatter(i, j, color='gold', edgecolor='black', s=250, zorder=5, marker='*', label='整数最优解')
#             else:
#                 plt.scatter(i, j, color='blue', s=60, zorder=4, label='可行整数解' if (i==1 and j==0) else "")
#         else:
#             plt.scatter(i, j, color='grey', s=30, alpha=0.5, zorder=3, label='不可行整数解' if (i==6 and j==4) else "")

# # 4. 绘制目标函数扫描线 Z = x1 + 3*x2
# plt.plot(x_val, 9/3 - x_val/3, color='orange', linestyle='--', linewidth=2.5, zorder=4, label='目标值Z=9 (整数域最大值)')
# plt.plot(x_val, 10/3 - x_val/3, color='darkred', linestyle='--', linewidth=2.5, zorder=4, label='目标值Z=10 (连续域最大值)')

# # 5. 图表排版与美化
# plt.xlim(-0.5, 6.5)
# plt.ylim(-0.5, 4.5)
# plt.xticks(range(7))
# plt.yticks(range(5))
# plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
# plt.xlabel('x1', fontsize=12)
# plt.ylabel('x2', fontsize=12)
# plt.title('连续可行域与整数间隙', fontsize=14, fontweight='bold')

# # 合并图例并调整位置，避免遮挡主要图形
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)

# plt.tight_layout()

# # 保存图片
# file_name = 'mip_gap_english_region.png'
# plt.savefig(file_name, dpi=300, bbox_inches='tight')
# print(f"绘图成功！请在当前目录下查看生成的图片文件: {file_name}")

# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 6.5), dpi=150)
# plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# # 坐标轴延伸范围，为了展示完整的直线切分
# x_val = np.linspace(-1, 7, 400)
# y_constraint = (10 - 2 * x_val) / 3

# # 1. 绘制核心约束的完整直线 (贯穿坐标系)
# plt.plot(x_val, y_constraint, color='red', linewidth=2, zorder=2, label='连续边界: 2x1 + 3x2 = 10')

# # 2. 严格填充连续可行域 (只在 x1>=0 且 x2>=0 且 满足约束 的区域填充)
# x_fill = np.linspace(0, 5, 200)
# y_fill = (10 - 2 * x_fill) / 3
# plt.fill_between(x_fill, 0, y_fill, color='red', alpha=0.15, zorder=1, label='连续可行域')

# # 加粗 X 轴和 Y 轴，强调第一象限的非负约束边界
# plt.axvline(x=0, color='black', linewidth=1.5, zorder=2)
# plt.axhline(y=0, color='black', linewidth=1.5, zorder=2)

# # 3. 绘制离散整数点
# for i in range(7):
#     for j in range(5):
#         if 2*i + 3*j <= 10:
#             if i == 0 and j == 3:
#                 # 整数最优解
#                 plt.scatter(i, j, color='gold', edgecolor='black', s=250, zorder=5, marker='*', label='整数最优解 (0, 3)')
#             else:
#                 plt.scatter(i, j, color='blue', s=60, zorder=4, label='可行整数解' if (i==1 and j==0) else "")
#         else:
#             plt.scatter(i, j, color='grey', s=30, alpha=0.5, zorder=3, label='不可行整数解' if (i==6 and j==4) else "")

# # 4. 绘制目标函数扫描线 Z = x1 + 3*x2
# plt.plot(x_val, 9/3 - x_val/3, color='orange', linestyle='--', linewidth=2.5, zorder=4, label='目标值Z=9 (整数域最大值)')
# plt.plot(x_val, 10/3 - x_val/3, color='darkred', linestyle='--', linewidth=2.5, zorder=4, label='目标值Z=10 (连续域最大值)')

# # ================= 新增：标记连续最优解与 Gap =================
# # 标记连续最优解点 (0, 10/3)
# plt.scatter(0, 10/3, color='darkred', edgecolor='black', s=120, zorder=6, marker='o', label='连续最优解 (0, 10/3)')

# # 画出表示 Gap 的线段 (Y轴上 3 到 10/3 的部分)
# plt.plot([0, 0], [3, 10/3], color='purple', linewidth=4, zorder=6, label='整数 Gap (Slack)')

# # 用箭头和文字将 Gap 注释出来，放在图形稍微靠右的位置避免重叠
# plt.annotate('整数 Gap\n(Slack)', 
#              xy=(0, 3.16),            # 箭头指向的坐标 (线段的中点)
#              xytext=(0.5, 3.5),       # 文字所在的坐标
#              arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, connectionstyle="arc3,rad=.2"),
#              fontsize=12, color='purple', fontweight='bold', zorder=7)
# # ==============================================================

# # 5. 图表排版与美化
# plt.xlim(-0.5, 6.5)
# plt.ylim(-0.5, 4.5)
# plt.xticks(range(7))
# plt.yticks(range(5))
# plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
# plt.xlabel('x1', fontsize=12)
# plt.ylabel('x2', fontsize=12)
# plt.title('连续可行域与整数间隙', fontsize=14, fontweight='bold')

# # 合并图例并调整位置，避免遮挡主要图形
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)

# plt.tight_layout()

# # 保存图片
# file_name = 'mip_gap_slack.png'
# plt.savefig(file_name, dpi=300, bbox_inches='tight')
# print(f"绘图成功！请在当前目录下查看生成的图片文件: {file_name}")



import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6.5), dpi=150)
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 坐标轴延伸范围，为了展示完整的直线切分
x_val = np.linspace(-1, 7, 400)
y_constraint = (10 - 2 * x_val) / 3

# 1. 绘制核心约束的完整直线 (贯穿坐标系)
plt.plot(x_val, y_constraint, color='red', linewidth=2, zorder=2, label='约束边界2x1+3x2=10')

# 2. 严格填充连续可行域 (只在 x1>=0 且 x2>=0 且 满足约束 的区域填充)
x_fill = np.linspace(0, 5, 200)
y_fill = (10 - 2 * x_fill) / 3
plt.fill_between(x_fill, 0, y_fill, color='red', alpha=0.15, zorder=1, label='连续可行域')

# 加粗 X 轴和 Y 轴，强调第一象限的非负约束边界
plt.axvline(x=0, color='black', linewidth=1.5, zorder=2)
plt.axhline(y=0, color='black', linewidth=1.5, zorder=2)

# 3. 绘制离散整数点
for i in range(7):
    for j in range(5):
        if 2*i + 3*j <= 10:
            if i == 0 and j == 3:
                # 整数最优解
                plt.scatter(i, j, color='gold', edgecolor='black', s=250, zorder=5, marker='*', label='整数最优解(0,3),Z=9')
                # 标记连续最优解点 (0, 10/3) 为绿色星星
                plt.scatter(0, 10/3, color='limegreen', edgecolor='black', s=250, zorder=6, marker='*', label='连续最优解(0,3.33)')
            else:
                plt.scatter(i, j, color='blue', s=60, zorder=4, label='整数可行解' if (i==1 and j==0) else "")
        else:
            plt.scatter(i, j, color='grey', s=30, alpha=0.5, zorder=3, label='不可行整数解' if (i==6 and j==4) else "")

# 4. 绘制目标函数扫描线 Z = x1 + 3*x2
plt.plot(x_val, 9/3 - x_val/3, color='orange', linestyle='--', linewidth=2.5, zorder=4, label='目标值Z=9(整数最大值)')
plt.plot(x_val, 10/3 - x_val/3, color='darkred', linestyle='--', linewidth=2.5, zorder=4, label='目标值Z=10(连续最大值)')

# ================= 标记连续最优解与 Slack =================


# 用黑字和黑色箭头将 Slack 注释出来
plt.annotate('Slack', 
             xy=(0, 3.16),            # 箭头指向的坐标 (两个星星中间)
             xytext=(0.5, 3.5),       # 文字所在的坐标
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle="arc3,rad=.2"),
             fontsize=12, color='black', fontweight='bold', zorder=7)
# ==========================================================

# 5. 图表排版与美化
plt.xlim(-0.5, 6.5)
plt.ylim(-0.5, 4.5)
plt.xticks(range(7))
plt.yticks(range(5))
plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
# plt.title('MIP: Continuous Feasible Region vs Integrality Gap', fontsize=14, fontweight='bold')

# 合并图例并调整位置，避免遮挡主要图形
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)

plt.tight_layout()

# 保存图片
file_name = 'mip_gap_slack_updated.png'
plt.savefig(file_name, dpi=300, bbox_inches='tight')
print(f"绘图成功！请在当前目录下查看生成的图片文件: {file_name}")