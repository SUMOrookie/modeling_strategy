# 基于约简策略的混合整数规划加速求解方法研究

> 毕业论文代码仓库 —— 两种基于机器学习与优化的 MIP 加速求解方法

## 项目概述

本课题研究**混合整数规划（MIP）的加速求解方法**，核心思想是利用图神经网络（GNN）预测约束的重要性，对问题施加约简（聚合或删除约束/变量），以更小的代价获得近似最优解。共包含两种方法：

### 方法一：基于替代约束的约束约简（对应论文第3章）
- 用 **EGAT（Edge Graph Attention Network）** 在二部图上预测每条约束的"紧度"分数
- 选取最重要的 `k` 条约束，用乘子法聚合成少量替代约束
- 短时间求解聚合后的简化问题，再通过**启发式修复**和**大邻域搜索**恢复可行性与最优性
- 对比方法：Gurobi 直接求解、PredictAndSearch（固定变量搜索）

### 方法二：对偶视角下的约束-变量协同约简（对应论文第4章）
- 在方法一基础上改进监督标签的构造方式（利用对偶信息）
- 新增**基于检验数（reduced cost）的变量约简**模块：在求解聚合问题后，根据松弛解的检验数筛选并固定部分变量，进一步缩小搜索空间
- 约束聚合阶段引入多种相似度度量（Jaccard、Cosine）和对偶归一化策略

---

## 文件说明

### 🔵 核心代码（论文方法实现）

| 文件 | 功能说明 |
|------|----------|
| `EGAT_layers.py` | EGAT 图注意力网络层定义，包括稀疏图注意力层 `SpGraphAttentionLayer` 和自定义稀疏反向传播 |
| `EGAT_models.py` | EGAT 模型主体 `SpGAT`：在约束-变量二部图上执行双向消息传递，输出每条约束的重要性分数 |
| `train.py` | **训练脚本**：加载二部图数据，训练 EGAT 模型预测约束紧度。包含 `Focal_Loss`（处理类别不平衡）、`GraphDataset` 数据加载器、训练/验证循环 |
| `gen_train_data.py` | **训练数据生成**：对每个 LP 实例，随机采样约束子集 → 聚合 → 求解 → 计算时间约简作为标签。支持多进程并行，输出约束子集及其时间收益 |
| `test.py` | **测试脚本（方法二）**：加载预训练 EGAT → 预测约束重要性 → 聚合 → 求解聚合问题 → 基于检验数约简变量 → 修复/邻域搜索后处理 → 输出性能指标 |
| `app.py` | **Streamlit Web 演示系统**："组合优化加速求解与评估系统"，集成了 Gurobi 求解、约束约简、协同约简三种策略的可视化对比 |
| `utils.py` | **核心工具库（约1600行）**：Gurobi/SCIP 回调函数、求解缓存管理、多种约束聚合方法（`aggregate_constr`、`aggregate_constr_two_two`、`aggregate_constr_duals`）、对偶信息处理、LP 松弛求解、相似度计算、贪婪配对聚合等 |
| `parser_utils.py` | 训练/测试的命令行参数解析器（学习率、隐藏层维度、注意力头数等超参数） |
| `agg_utils.py` | 聚合求解辅助函数：获取聚合约束列表、求解聚合实例、处理变量固定（检验数筛选） |
| `repair_and_post_solve_func.py` | **修复与后处理**：6种修复策略（naive/score/subproblem/subproblem_cons_geq/lightmilp）、大邻域搜索（`PostSolve`）、SCIP 版本的修复函数 |
| `get_bigraph.py` | 从 Gurobi 模型提取变量-约束二部图（节点特征、边特征、邻接矩阵），供 GNN 输入 |
| `cal_var_Reduce_num.py` | **变量约简模块**：计算哪些变量可被固定，统计约简的变量数量，评估约简效果 |
| `exp_compare_solution_distance.py` | **实验对比脚本**：对比不同方法（Gurobi / PredictAndSearch / 本文方法）的解距离（Manhattan distance）和求解性能 |

### 🟡 辅助与实验脚本

| 文件 | 功能说明 |
|------|----------|
| `draw.py` | 绘图工具（目前已注释），用于绘制连续可行域与整数间隙的示意图 |
| `dual_info.py` | **几乎为空**（仅一个函数签名），原本计划用于提取对偶信息，实际未完成 |
| `parameters/` | 实验超参数 JSON 配置文件（`param_0.json` ~ `param_large.json` 等），对应不同实验设置 |

### 🔴 临时/废弃文件（研究过程中产生，论文不需要）

| 文件 | 说明 | 原因 |
|------|------|------|
| `demo.py` | 一个简单的整数规划小例子（x1~x4 手动建模求解） | 早期学习 Gurobi API 的测试代码，与论文无关 |
| `demo1.py` | 类似 `test.py` 的早期版本，调用 `accelerated_solving` | 功能已被 `test.py` 覆盖，是开发过程中的过渡版本 |
| `main.py` | 早期的实验入口，使用 SCIP 和素数乘子做聚合求解 | 被 `test.py` 替代，仅保留作参考 |
| `generate_instance.py` | 使用 ecole/SCIP 生成 MVC/CA/IS/SC 问题实例 | 新版，可能仍可用于生成新实例 |
| `generateInstance.py` | 与 `generate_instance.py` 几乎相同 | 旧版，功能重复 |
| `gen_general_CA.py` | 生成含 ≤ 和 ≥ 约束的广义组合拍卖问题，并测试可行性修复 | 特定实验使用，非常规流程 |
| `gen_assignment_problem.py` | 生成指派问题 LP 文件 | 基本用不上，仅用于早期测试 |
| `test_random_cons.py` | 随机选择约束聚合（不经过 GNN 预测），对比各种修复函数效果 | 消融实验脚本，验证 GNN 预测是否优于随机选择 |
| `test_random_cons_scip.py` | 同上，使用 SCIP 求解器版本 | 消融实验 SCIP 版本 |
| `test_random_solutions_distance.py` | 随机聚合 + 计算解距离 | 早期探索性实验 |

### 🟢 predictandsearch/ — 对比基线方法

这是论文的对比方法 **PredictAndSearch**（来自 Nair et al., "Solving Mixed Integer Programs Using Neural Networks"），完整保留了原始框架代码：

| 文件 | 功能说明 |
|------|----------|
| `predictandsearch/GCN.py` | PredictAndSearch 的 GCN 模型（`GNNPolicy`）和数据集加载（`GraphDataset`），使用 PyTorch Geometric |
| `predictandsearch/trainPredictModel.py` | PredictAndSearch 的训练脚本 |
| `predictandsearch/PredictAndSearch_GRB.py` | **Gurobi 版 PredictAndSearch**：加载预训练 GCN → 预测变量取值 → 固定部分变量 → Gurobi 求解 |
| `predictandsearch/PredictAndSearch_SCIP.py` | **SCIP 版 PredictAndSearch**：同上流程，使用 SCIP 求解器 |
| `predictandsearch/FixingStrategy_SCIP.py` | SCIP 版变量固定策略实现 |
| `predictandsearch/helper.py` | 辅助函数：二部图提取（SCIP 版本）、位置编码、数据预处理 |
| `predictandsearch/gurobi.py` | Gurobi 多解求解工具：生成训练数据用的多解池（solution pool） |

---

## 目录结构

```
modeling_strategy/
├── EGAT_layers.py              # GNN 注意力层
├── EGAT_models.py              # GNN 模型定义
├── train.py                    # 训练 GNN
├── gen_train_data.py           # 生成训练数据（标签构造）
├── test.py                     # 测试/推理（方法二主入口）
├── app.py                      # Streamlit Web 演示
├── utils.py                    # 核心工具库（约束聚合、缓存、对偶）
├── parser_utils.py             # 参数解析
├── agg_utils.py                # 聚合求解辅助
├── repair_and_post_solve_func.py  # 修复与邻域搜索
├── get_bigraph.py              # 二部图提取
├── cal_var_Reduce_num.py       # 变量约简统计
├── exp_compare_solution_distance.py  # 实验对比
├── draw.py                     # 绘图工具
├── dual_info.py                # （未完成）
│
├── demo.py                     # [废弃] Gurobi 学习测试
├── demo1.py                    # [废弃] 早期测试版本
├── main.py                     # [废弃] 早期实验入口
├── generate_instance.py        # [可能使用] 实例生成
├── generateInstance.py         # [废弃] 旧版实例生成
├── gen_general_CA.py           # [特殊用途] 广义CA生成
├── gen_assignment_problem.py   # [废弃] 指派问题生成
├── test_random_cons.py         # [消融实验] 随机聚合对比
├── test_random_cons_scip.py    # [消融实验] SCIP版随机聚合
├── test_random_solutions_distance.py  # [废弃] 早期实验
│
├── predictandsearch/           # 对比基线方法（PredictAndSearch）
│   ├── GCN.py
│   ├── trainPredictModel.py
│   ├── PredictAndSearch_GRB.py
│   ├── PredictAndSearch_SCIP.py
│   ├── FixingStrategy_SCIP.py
│   ├── helper.py
│   └── gurobi.py
│
├── parameters/                 # 超参数配置 JSON
├── dataset/                    # 训练/测试数据集（二部图、标签等）
├── instance/                   # LP 问题实例
├── model/                      # 训练好的模型权重
├── cache/                      # 求解缓存
├── log/                        # 实验日志
├── parsed/                     # 解析后的实例数据
├── relax_results/              # 松弛求解结果
├── result/                     # 测试结果输出
└── .vscode/                    # VS Code 配置
```

## 环境配置

环境依赖见 `environment.yml`，主要依赖：
- **Python 3.8+**
- **Gurobi**（商业求解器，需 license）
- **PyTorch** + **PyTorch Geometric**（图神经网络）
- **PySCIPOpt**（SCIP 求解器，用于部分基线实验）
- **Streamlit**（Web 演示）
- **ecole**（组合优化环境，用于实例生成）

## 运行流程概要

### 方法一（约束约简）
1. `generate_instance.py` → 生成 LP 实例
2. `gen_train_data.py` → 采样约束子集、聚合求解，生成训练数据
3. `get_bigraph.py` + `EGAT_models.py` → 从 LP 提取二部图
4. `train.py` → 训练 EGAT 预测约束重要性
5. `test.py`（`fix_var=False`）→ 推理：GNN 预测 → 聚合 → 短时间求解 → 修复 → 邻域搜索

### 方法二（协同约简）
1. 同方法一步骤 1-4
2. `test.py`（`fix_var=True`）→ 在方法一基础上增加检验数变量固定步骤
3. `cal_var_Reduce_num.py` → 评估变量约简效果

### 实验对比
- `exp_compare_solution_distance.py` → 对比 Gurobi / PredictAndSearch / 本文方法
- `predictandsearch/PredictAndSearch_GRB.py` → 运行基线方法

---

## 关键机制详解

### 一、训练标签的两种构造方式

GNN 的训练目标是预测每条约束的"重要性"（是否应该被聚合）。代码中实现了两种标签构造策略，在 `gen_train_data.py` 中生成原始数据，在 `train.py` 中切换使用：

#### 方式 1：随机采样 + 最优子集（对应模型 `model_random_sample.pth`）

**原理**：对每个训练实例，随机采样大量约束子集 → 对每个子集执行聚合求解 → 找到使聚合解距离最优解**最近**的那个子集 → 该子集中的约束标记为重要（label=1），其余为不重要（label=0）。

**代码位置**：[train.py:220-229](train.py#L220-L229)（当前被注释）：

```python
# 2.最好的子集(距离最优解距离最近)
constr_label = [0 for idx in range(m)]
best_subset_idx = -1
best_agg_time = 1e10
random_sample = solve_info["random_set"]
random_sample.sort(key=lambda x: x["distance"])
best_constr_set = random_sample[0]["constr_set"]
for idx in best_constr_set:
    constr_label[idx] = 1
```

**数据来源**：`gen_train_data.py` 中的 `gen_constr_label()` 函数（第 192 行）随机采样约束子集，`_gen_constr_label_worker()`（第 27 行）对每个子集求解聚合问题并计算与最优解的 distance，结果存入 `solve_info['random_set']`。

#### 方式 2：最优解处的约束松弛余量（对应模型 `model_tight_constr.pth`）

**原理**：求解原始问题的 LP 松弛或直接获取最优解处的约束 slack 值。**slack = 0 的约束是"紧约束"**（在最优解处取等号，对界定最优解至关重要，保留），**slack > 0 的约束是"松弛约束"**（有冗余空间，可聚合，label=1）。

**代码位置**：[train.py:236-239](train.py#L236-L239)（**当前默认激活**）：

```python
# 4.用约束的slack作为标签
# slack=0为紧约束，对应标签为0，slack>0为松弛约束，对应标签为1。标签为1的约束进行聚合。
constr_label = [0 if solve_info['slack'][idx][1]==0 else 1 for idx in range(m)]
```

**数据来源**：slack 值来自求解缓存（`cache[lp_path]['slack']`），在 `gen_train_data.py` 第 499 行一并存入 `solve_info`。

#### 切换方式

在 `test.py` 第 129-130 行，通过加载不同的模型权重来对应两种标签：

```python
# CHECKPOINT_PATH = f"./model/{task_name}/model_tight_constr.pth"   # 方式2: slack标签
CHECKPOINT_PATH = f"./model/{task_name}/model_random_sample.pth"    # 方式1: 随机采样子集标签
```

同时 `train.py` 中第 206-239 行的注释块也需对应切换，确保训练和推理使用一致的标签定义。

> 此外，`train.py` 中还保留了其他未使用的标签方案（已注释）：对偶变量标签（第 233-234 行）、时间约简分数标签（第 210-217 行）。

---

### 二、`fix_var`：是否启用基于检验数的变量约简

`fix_var` 是 `test.py` 中的布尔变量（[test.py:149-150](test.py#L149-L150)），控制**是否在求解聚合问题后执行变量约简**（即论文方法二的核心新增模块）：

```python
# fix_var = False   # 方法一：仅约束约简，不固定变量
fix_var = True      # 方法二：约束约简 + 检验数变量约简
```

其具体逻辑在 [`agg_utils.py:16-82`](agg_utils.py) 的 `solve_agg_instance()` 函数中：

#### `fix_var = True` 的执行流程：

1. **求解聚合问题的整数版本**：复制模型 → 在 `agg_model_solve_time` 时限内求整数解 → 得到 `vaule_dict`
2. **求解聚合问题的 LP 松弛**：将原聚合模型的所有变量类型改为连续 → 求解松弛 → 获取每个变量的 **reduced cost**
3. **计算阈值并固定变量**：
   - 计算所有负检验数的**平均值**作为阈值
   - 将 `RC ≤ threshold` 的变量固定为其在整数解中的值（通常为 0）
   - 统计被额外固定的变量数量
4. **返回**：固定后的解 `vaule_dict` 和检验数列表（供后续邻域搜索使用）

```python
# agg_utils.py 第 57-66 行
if sum([1 if r < 0 else 0 for r in reduced_costs]) != 0:
    threshold = sum([r if r < 0 else 0 for r in reduced_costs]) / sum([1 if r < 0 else 0 for r in reduced_costs])
    fixed_vars = [v.VarName for v in model_agg.getVars() if v.RC <= threshold]
    for varname in fixed_vars:
        if vaule_dict[varname] != 0:
            cnt+=1
        vaule_dict[varname] = 0
```

#### `fix_var = False` 的执行流程：

直接求解聚合问题，返回变量值，**不进行任何变量固定**。

#### 检验数的直观含义

在 LP 松弛的最优解处，变量的 **reduced cost** 表示该变量进入基（从 0 变为正数）所需的目标函数"代价"。若某变量在松弛解中为 0 且其 reduced cost 的绝对值很大（很负），说明该变量不太可能出现在最优整数解中，可以安全地固定为 0。这就是方法二中"变量约简"的理论依据。

## License

见 [LICENSE](LICENSE)
