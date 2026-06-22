from __future__ import division
from __future__ import print_function

import os
import glob
import time
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import parser_utils
from EGAT_models import SpGAT
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,roc_auc_score,average_precision_score

def plot_and_save_loss(loss_values: List[torch.Tensor], output_path: str = "loss_curve.png") -> None:
    """
    绘制损失值曲线并保存为图片

    参数:
    - loss_values: 包含Tensor类型损失值的列表
    - output_path: 图片保存路径
    """
    # 从Tensor中提取数值
    loss_values = [float(loss) for loss in loss_values]

    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, 'b-o', linewidth=2, markersize=6)

    # 添加标题和标签
    plt.title('训练损失值变化曲线', fontsize=16)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('损失值', fontsize=14)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置坐标轴范围
    plt.ylim(bottom=0)

    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"损失曲线已保存至: {output_path}")

    # 关闭图形
    plt.close()
class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # List in tensor data format

    def forward(self, preds, labels):
        """
        preds: logits output values
        labels: labels
        """
        # preds = F.softmax(preds, dim=1).to(device) # 为什么要在做一次？
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels).to(device)
        ce = (-1 * torch.log(preds + eps) * target).to(device)
        floss = (torch.pow((1 - preds), self.gamma) * ce).to(device)
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one



parser = parser_utils.get_parser("train")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


data_features = []
data_labels = []
data_solution = []
data_edge_features = []
data_edge_A = []
data_edge_num_A = []
data_edge_B = []
data_edge_num_B = []
data_idx_train = []

# task_name = "CA_750_1100_0.7"
# task_name = "CA_650_1000_0.7"
task_name = "IS_1500_6"
dataset_dir = f"./dataset/{task_name}"
BG_dir = os.path.join(dataset_dir,"BG")
constr_score_dir = os.path.join(dataset_dir,"constr_score")


# dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pickle')]
BG_files = [f for f in os.listdir(BG_dir) if f.endswith('.pickle')]
# constr_score_files = [f for f in os.listdir(constr_score_dir) if f.endswith('.pickle')]

BG_files.sort()
# constr_score_files.sort()


data_num = len(BG_files)

# random.shuffle(dataset_files)
threshold = 0
# k = 50
# for pickle_file_name in dataset_files[:data_num]:

solve_info_dir = os.path.join(dataset_dir,"solve_info")
solve_info_files = [f for f in os.listdir(solve_info_dir) if f.endswith('.pickle')]
solve_info_files.sort()
for pickle_file_name,solve_info_file_name in zip(BG_files,solve_info_files):
    pickle_path = os.path.join(BG_dir,pickle_file_name)
    # constr_score_path = os.path.join(constr_score_dir,constr_score_file_name)
    solve_info_path = os.path.join(solve_info_dir, solve_info_file_name)
    with open(solve_info_path, "rb") as f:
        solve_info = pickle.load(f)
    with open(pickle_path,"rb") as f:
        problem = pickle.load(f)
    # with open(constr_score_path,"rb") as f:
    #     constr_score = pickle.load(f)


    variable_features = problem[0]
    constraint_features = problem[1]
    edge_indices = problem[2]
    edge_feature = problem[3]
    # constr_score = constr_score[0]
    #print(optimal_solution)
    #edge, features, labels, idx_train = load_data()

    #change
    n = len(variable_features)
    var_size = len(variable_features[0])
    m = len(constraint_features)
    con_size = len(constraint_features[0])

    edge_num = len(edge_indices[0])
    data_edge_num_A.append(edge_num)
    edge_num = len(edge_indices[0])
    data_edge_num_B.append(edge_num)

    edgeA = []
    edgeB = []
    edge_features = []
    for i in range(edge_num):
        edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
        edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
        edge_features.append(edge_feature[i])
    edgeA = torch.as_tensor(edgeA)
    data_edge_A.append(edgeA)

    edgeB = torch.as_tensor(edgeB)
    data_edge_B.append(edgeB)

    edge_features = torch.as_tensor(edge_features)
    data_edge_features.append(edge_features)

    if var_size > con_size:
        for i in range(m):
            for j in range(var_size - con_size):
                constraint_features[i].append(0)
    else:
        for i in range(n):
            for j in range(con_size - var_size):
                variable_features[i].append(0)
    features = variable_features + constraint_features
    features = torch.as_tensor(features).float()

    # 加入这两行，用来保证没有 NaN/Inf
    # assert not torch.isnan(features).any(), f"NaN in features of instance "
    # assert not torch.isinf(features).any(), f"Inf  in features of instance "
    # assert not torch.isnan(edge_features).any(), f"NaN in edge_feat of instance "
    # assert not torch.isinf(edge_features).any(), f"Inf  in edge_feat of instance"

    data_features.append(features)

    # 得到constr_label
    ## 可以有多种方式
    # 1.变量分数平均值
    # constr_label = [0 for idx in range(m)]
    # cnt = 0
    # for pair in constr_score:
    #     constr_idx = pair[0]
    #     time_reduce = pair[1]
    #     if cnt < k and time_reduce > threshold:
    #         constr_label[constr_idx] = 1
    #         cnt += 1

    # 2.最好的子集(距离最优解距离最近)
    # constr_label = [0 for idx in range(m)]
    # cnt = 0
    # # 找到最好的子集
    # best_subset_idx = -1
    # best_agg_time = 1e10
    # random_sample = solve_info["random_set"]
    # random_sample.sort(key=lambda x:x["distance"])
    # best_constr_set = random_sample[0]["constr_set"]
    # for idx in best_constr_set:
    #     constr_label[idx] = 1

    # 3.用松弛问题的对偶变量作为标签
    # 暂时先用对偶值为0的标签为0，对偶值为1的标签为1的方式来做吧。
    # constr_label = [0 for idx in range(m)]
    # constr_label = [1 if dual_var == 0 else 0 for dual_var in solve_info['relaxed_dual_solution']]

    # 4.用约束的slack作为标签
    # slack=0为紧约束，对应标签为0，slack>0为松弛约束，对应标签为1。标签为1的约束进行聚合。
    # 这里还没有考虑阈值的情况
    constr_label = [0 if solve_info['slack'][idx][1]==0 else 1 for idx in range(m)]

    # focal loss的权重
    # num_label = [1, 20]
    num_label = [1, 1]
    num_label = torch.as_tensor(num_label).to(device)
    data_labels.append(num_label)

    # 加入变量的label
    constr_label = [0 for _ in range(n)] + constr_label

    labels = torch.as_tensor(constr_label)

    data_solution.append(labels)

    # 需要计算loss的部分，也就是约束的范围
    idx_train = torch.as_tensor(range(n,n+m))
    data_idx_train.append(idx_train)




# Model and optimizer
model = SpGAT(nfeat=data_features[0].shape[1],    # Feature dimension
            nhid=args.hidden,             # Feature dimension of each hidden layer
            nclass=int(data_solution[0].max()) + 1, # Number of classes
            dropout=args.dropout,         # Dropout
            nheads=args.nb_heads,         # Number of heads
            alpha=args.alpha)             # LeakyReLU alpha coefficient

class GraphDataset(Dataset):
    def __init__(self, data_features, data_labels, data_solution,
                 data_edge_A, data_edge_B, data_edge_features,
                 data_idx_train):
        # 存储所有数据的列表。注意：这里假设传入的已经是张量列表
        self.features = data_features
        self.labels = data_labels
        self.solutions = data_solution
        self.edge_A = data_edge_A
        self.edge_B = data_edge_B
        self.edge_features = data_edge_features
        self.idx_train = data_idx_train

        # 确保所有数据的数量一致
        assert len(self.features) == len(self.labels) == len(self.solutions)

    def __len__(self):
        # 数据集中的样本总数 (即您的 data_num)
        return len(self.features)

    def __getitem__(self, idx):
        # 根据索引 idx 返回一组数据 (对应一个图)
        sample = {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'solution': self.solutions[idx],
            'edge_A': self.edge_A[idx],
            'edge_B': self.edge_B[idx],
            'edge_features': self.edge_features[idx],
            'idx_train': self.idx_train[idx]
        }
        return sample

optimizer = optim.Adam(model.parameters(),    
                       lr=args.lr,                        # Learning rate
                       weight_decay=args.weight_decay)    # Weight decay to prevent overfitting

num_graphs = len(data_features)
all_graph_indices = list(range(num_graphs))
train_graph_indices, val_graph_indices = train_test_split(
    all_graph_indices,
    test_size=0.2, # 20% 作为验证集
    random_state=42 # 保证可复现
)
print(f"总图数: {num_graphs}")
print(f"训练图数: {len(train_graph_indices)}")
print(f"验证图数: {len(val_graph_indices)}")


full_dataset = GraphDataset(data_features, data_labels, data_solution,
                            data_edge_A, data_edge_B, data_edge_features,
                            data_idx_train)
train_dataset = Subset(full_dataset, train_graph_indices)
val_dataset = Subset(full_dataset, val_graph_indices)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True, # 训练集需要打乱
    num_workers=0
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False, # 验证集不需要打乱
    num_workers=0
)
# 实例化 Dataset
# dataset = GraphDataset(data_features, data_labels, data_solution,
#                        data_edge_A, data_edge_B, data_edge_features,
#                        data_idx_train)
#
#
# # 实例化 DataLoader
# # shuffle=True 可以在每次循环时实现随机抽取
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=0 # GNN 数据通常不适合多进程 (num_workers > 0)
# )


def train(model, optimizer, data_batch, device):
    """
    接收一个数据批次（在我们的例子中是一个图样本）进行训练。
    """
    # 1. 设备转移：现在，我们只在需要时将数据转移到 GPU 上
    features = data_batch['features'][0].to(device)  # [0] 是因为 batch_size=1 会增加一个维度
    edge_A = data_batch['edge_A'][0].to(device)
    edge_B = data_batch['edge_B'][0].to(device)
    edge_features = data_batch['edge_features'][0].to(device)
    solution = data_batch['solution'][0].to(device)
    idx_train = data_batch['idx_train'][0].to(device)
    labels = data_batch['labels'][0]  # 保持在 CPU，如果 Focal_Loss 需要 CPU

    # 2. 前向传播
    model.train()
    optimizer.zero_grad()

    output, new_edge_features = model(features, edge_A, edge_B, edge_features.detach())

    # 3. 计算损失
    # 确保 Focal_Loss 的输入张量设备一致
    lf = Focal_Loss(torch.as_tensor(labels).to(device))  # 确保 labels 也转移到 device
    loss_train = lf(output[idx_train], solution[idx_train])

    # 4. 反向传播和优化
    loss_train.backward()
    optimizer.step()

    return loss_train.item()


def validate(model, data_batch, device):
    """
    接收一个数据批次进行验证，计算损失和原始输出。
    """
    # 1. 设备转移
    features = data_batch['features'][0].to(device)
    edge_A = data_batch['edge_A'][0].to(device)
    edge_B = data_batch['edge_B'][0].to(device)
    edge_features = data_batch['edge_features'][0].to(device)
    solution = data_batch['solution'][0].to(device)
    idx_nodes = data_batch['idx_train'][0].to(device)
    labels = data_batch['labels'][0]

    # 2. 前向传播
    model.eval()
    with torch.no_grad():
        output, _ = model(features, edge_A, edge_B, edge_features.detach())

        # 3. 计算损失
        lf = Focal_Loss(torch.as_tensor(labels).to(device))
        loss_val = lf(output[idx_nodes], solution[idx_nodes])

        # 4. 提取目标的 Logits 和 真实的 Targets
        logits_cpu = output[idx_nodes].cpu().numpy()
        targets_cpu = solution[idx_nodes].cpu().numpy()

    # 🌟 返回损失、原始 Logits 和 目标
    return loss_val.item(), logits_cpu, targets_cpu


# 确保在训练前模型已在目标设备上
if args.cuda:
    model.to(device)

# ... [Dataset 和 DataLoader 实例化] ...

t_total = time.time()
loss_values = []
val_loss_values = []
bad_counter = 0
best_loss = 1e3
best_val_loss = 1e3
best_epoch = 0

model_save_path = f"./model/{task_name}"
os.makedirs(model_save_path, exist_ok=True)

now_time = time.time()
for epoch in range(args.epochs):
    print(f"--- Epoch: {epoch+1}/{args.epochs} ---")
    t_epoch_start = time.time()

    ## 训练
    model.train()
    epoch_train_loss = 0
    num_batches = 0
    for i, data_batch in enumerate(train_dataloader):
        batch_loss = train(model, optimizer, data_batch, device)
        epoch_train_loss += batch_loss
        num_batches += 1

    avg_train_loss = epoch_train_loss / num_batches

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(avg_train_loss))

    ## 验证
    model.eval()  # 切换到评估模式
    epoch_val_loss = 0
    num_val_batches = 0
    all_logits = []
    all_targets = []

    for i, data_batch in enumerate(val_dataloader):
        batch_loss, logits_cpu, targets_cpu = validate(model, data_batch, device)
        epoch_val_loss += batch_loss
        num_val_batches += 1

        # 收集所有批次 (图) 的预测和标签
        all_logits.append(logits_cpu)
        all_targets.append(targets_cpu)

    avg_val_loss = epoch_val_loss / num_val_batches
    # 汇总所有验证集的预测和标签
    # 因为每个图的节点数不同，使用 np.hstack
    all_logits = np.vstack(all_logits)
    all_targets = np.hstack(all_targets)

    # 1. 计算概率 (用于 AUC)
    # 我们需要对 Logits 应用 Softmax 得到概率
    # all_probs 是 (N, num_classes)
    all_probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()

    # 2. 计算预测类别 (用于 Acc, F1等)
    # all_preds 是 (N,)
    all_preds = np.argmax(all_logits, axis=1)

    # 3. 确定类别数量 (用于多分类 AUC)
    num_classes = all_logits.shape[1]

    # --- 核心指标计算 ---

    # 3a. 计算 Accuracy, F1 (Macro/Weighted)
    val_accuracy = accuracy_score(all_targets, all_preds)
    # Macro F1: 平等对待所有类别
    val_f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    # Weighted F1: 考虑类别不平衡
    val_f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    # 3b. 计算 AUC-ROC
    if num_classes == 2:
        # 二分类：只需提供正类的概率 all_probs[:, 1]
        val_auc_roc = roc_auc_score(all_targets, all_probs[:, 1])
        val_auc_pr = average_precision_score(all_targets, all_probs[:, 1])
    else:
        # 多分类：需要提供所有类的概率 all_probs
        # 使用 'ovr' (One-vs-Rest) 策略
        val_auc_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        # AUC-PR (average_precision_score) 对多分类支持不完善，
        # 我们通常看 'classification_report' 中的 F1/Recall/Precision
        val_auc_pr = -1  # 标记为不计算或使用更复杂的方法

    # 3c. 打印完整的分类报告
    print(f"\n--- Validation Report (Epoch {epoch + 1}) ---")
    print(classification_report(all_targets, all_preds, zero_division=0))
    print("--------------------------------------")

    # 记录损失
    loss_values.append(avg_train_loss)
    val_loss_values.append(avg_val_loss)

    print(f'Epoch: {epoch + 1:04d}',
          f'loss_train: {avg_train_loss:.4f}',
          f'loss_val: {avg_val_loss:.4f}',
          f'val_acc: {val_accuracy:.4f}',
          f'val_f1_macro: {val_f1_macro:.4f}',  # 推荐
          f'val_auc_roc: {val_auc_roc:.4f}',  # 推荐
          f'time: {time.time() - t_epoch_start:.4f}s')

    ## 3. 保存最佳模型 (基于验证集损失)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        bad_counter = 0

        filename = 'model_tight_constr.pth'
        # filename = 'model_tight_constr.pth'
        checkpoint_path = os.path.join(model_save_path, filename)

        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
    else:
        bad_counter += 1

    if bad_counter >= args.patience:
        print(f"Early stopping at epoch {epoch + 1}!")
        break

    # if loss_values[-1] < best_loss:
    #     best_loss = loss_values[-1]
    #     best_epoch = epoch
    #     bad_counter = 0
    #     # 2. 构造文件名
    #     # 使用 .pth 扩展名是 PyTorch 的常见做法
    #     # filename = f'best_model_run{now_time}_loss{best_loss:.4f}_epoch{best_epoch}.pth'
    #     filename = f'model.pth'
    #     checkpoint_path = os.path.join(model_save_path, filename)
    #
    #     # 3. 🌟 保存完整的 Checkpoint 🌟
    #     # 最佳实践是保存一个字典 (Dictionary)，包含模型、优化器和 epoch 信息
    #     torch.save({
    #         'epoch': best_epoch,
    #         'model_state_dict': model.state_dict(),  # 仅保存模型的参数
    #         'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器状态，以便断点恢复训练
    #         'best_loss': best_loss,
    #     }, checkpoint_path)
    # else:
    #     bad_counter += 1
    #
    # if bad_counter >= args.patience:  # Stop if there's no improvement for several consecutive rounds
    #     break

print("Optimization Finished!")
print(f"Total time elapsed: {time.time() - t_total:.4f}s")
print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

# 绘制损失图 (现在可以同时绘制训练和验证损失)
# plot_and_save_loss(loss_values, val_loss_values, "./loss.png")
plot_and_save_loss(loss_values,"./train_loss.png")




