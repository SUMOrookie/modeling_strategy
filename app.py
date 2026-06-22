import streamlit as st
import time
import numpy as np
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gurobipy as gp
import get_bigraph
import parser_utils
from EGAT_models import SpGAT
import agg_utils
import utils
import repair_and_post_solve_func
from gurobipy import GRB
import atexit
import shutil
import os
import tempfile
# ================= 新增：全局临时文件管理 =================
# 1. 在系统的 temp 目录下为本应用创建一个专属的隔离文件夹
APP_TMP_DIR = os.path.join(tempfile.gettempdir(), "opt_solver_tmp")
os.makedirs(APP_TMP_DIR, exist_ok=True)

# 2. 定义自毁钩子：不管程序怎么结束，把这个文件夹彻底删掉
def cleanup_temp_files():
    if os.path.exists(APP_TMP_DIR):
        shutil.rmtree(APP_TMP_DIR, ignore_errors=True)
        print("🧹 临时文件夹已清理完毕！")

# 3. 将其注册到 Python 解释器的退出生命周期中
atexit.register(cleanup_temp_files)
# ========================================================


# 1. 设置页面全局配置
st.set_page_config(
    page_title="组合优化加速求解系统",
    page_icon="🚀",
    layout="wide"
)

# 2. 主页面标题
st.title("组合优化加速求解与评估系统")
st.markdown("---")
st.markdown("**基于约简策略的组合优化问题加速求解方法研究 - 算法工程化演示平台**")

# 3. 左侧边栏：策略与参数配置 (对应论文 5.2.2 节)
with st.sidebar:
    st.header("⚙️ 求解策略与参数配置")
    
    # 策略选择分支
    strategy = st.selectbox(
        "选择加速求解策略", 
        [
            "使用Gurobi求解", 
            "基于替代松弛的约束约简", 
            "对偶视角下约束-变量协同约简"
        ]
    )
    
    st.divider()
    
    # 邻域搜索参数 (对应论文实验的 k0, k1, delta)
    st.subheader("大邻域搜索参数")
    # 将原本的 slider 改为了 number_input
    k0 = st.number_input("k0", min_value=0, value=400, step=10)
    k1 = st.number_input("k1", min_value=0, value=30, step=1)
    delta = st.number_input("delta", min_value=0, value=50, step=10)
    
    st.divider()
    
    # 新增参数：聚合问题求解时间与聚合约束数量 (参考论文3.6.2节)
    st.subheader("约简模型参数")
    agg_constraint_num = st.number_input(
        "聚合约束数量", 
        min_value=1, 
        value=50, 
        step=10, 
        help="组合拍卖(CA)实验中设为50，独立集(IS)设为300"
    )
    agg_time_limit = st.number_input(
        "聚合问题求解时间 (秒)", 
        min_value=1, 
        value=5, 
        step=1, 
        help="求解约简模型的限制时间，论文实验设定为5秒"
    )
    time_limit = st.number_input("全局求解时间上限 (秒)", min_value=1, value=1000, step=100)
    
    st.divider()
    st.info("💡 提示：后台将基于选择的策略进行相应求解")


# ================= 核心算法流占位函数 =================
def run_algorithm_pipeline(instance_data, model_path,strategy, params):
    """
    统筹调度深度学习与求解器的核心管道 (对应论文 5.2 节)
    返回绘图所需的日志数据
    """
    logs = {
        "time": [], 
        "objective": [], 
        "gap": [], 
        "original_scale": (0,0), 
        "reduced_scale": (0,0),
        "total_time": 0.0,
        "final_obj": float('nan'),
        "mip_gap": float('nan')
    }
    tmp_file_path = instance_data
    # 获取前端传进来的图表占位符
    chart_loc = params.get('chart_loc')
    # 记录上一次刷新网页的时间（使用列表作为可变对象，方便在闭包中修改）
    last_ui_update = [0.0]
    # ================= 新增：定义闭包 Callback 函数 =================
    def optimization_callback(model, where):
        is_updated = False
        
        # 统一处理 MIPSOL 和 MIP 节点的情况
        if where == GRB.Callback.MIPSOL or where == GRB.Callback.MIP:
            if where == GRB.Callback.MIPSOL:
                current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                current_bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            else:
                current_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                current_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                
            # 🔥 严防死守：过滤掉 1e100 这类无穷大毒数据
            if abs(current_obj) < 1e20 and abs(current_bnd) < 1e20:
                # 按照 Gurobi 官方逻辑计算 Gap，分母加 1e-10 防止除以 0
                gap_val = abs(current_obj - current_bnd) / max(1e-10, abs(current_obj))
                gap_pct = gap_val * 100.0 # 转换为百分比
                
                # 如果 Gap 发生变化，或者列表为空，则记录
                if len(logs["gap"]) == 0 or gap_pct != logs["gap"][-1]:
                    current_time = time.perf_counter() - t0
                    logs["time"].append(current_time)
                    logs["objective"].append(current_obj) # 依然保留原目标值备用
                    logs["gap"].append(gap_pct)           # 记录 Gap 百分比
                    is_updated = True

        # 🔥 实时绘图逻辑 (改画 Gap)
        if is_updated and chart_loc is not None:
            now = time.perf_counter()
            if now - last_ui_update[0] > 0.5:
                fig = go.Figure()
                
                # 画 Gap 曲线
                fig.add_trace(go.Scatter(
                    x=logs["time"], y=logs["gap"],
                    mode='lines+markers', name="实时 Gap (%)", 
                    line=dict(color='teal', width=3, shape='hv') # 换个好看的青色
                ))
                
                fig.update_layout(
                    title="🔍 寻优轨迹 (MIP Gap 实时收敛中...)",
                    xaxis_title="求解时间 (秒)",
                    yaxis_title="MIP Gap (%)",
                  yaxis=dict(
                    type="log",
                    exponentformat="power" # 完美显示 10^2, 10^1, 10^-1
                ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=300
                )
                chart_loc.plotly_chart(fig, use_container_width=True)
                last_ui_update[0] = now


    if instance_data is None or not isinstance(instance_data, str) or not os.path.exists(instance_data):
        st.error("找不到模型文件，请在左侧重新上传或生成实例！")
        return logs
        
    try:
        # 统一定义算法的起始时间戳（非常重要：后面的 callback 和耗时统计都要依赖它）
        t0 = time.perf_counter()
        
        # 3. 让 Gurobi 直接读取这个绝对物理路径
        model = gp.read(tmp_file_path)
        st.write(f"tmp_file_path: {tmp_file_path}")
        st.success("✅ 模型读取与特征提取准备完毕！")
        orig_vars = model.NumVars
        orig_constrs = model.NumConstrs
        st.write(f"📊 原始模型规模 -> 变量数量: {orig_vars} | 约束数量: {orig_constrs}")
        
        # 4. 正确记录原始规模到 logs 中
        logs["original_scale"] = (orig_vars, orig_constrs)
        
    except Exception as e:
        st.error(f"❌ Gurobi 读取模型失败: {e}")
        return logs
            
    logs["original_scale"] = (orig_vars, orig_constrs)

    t0 = time.perf_counter()  # 记录求解开始时间
    if strategy == "使用Gurobi求解":
        # TODO 2: 直接调用 Gurobi 求解，注册 Callback 记录时间与目标值
        model.setParam('TimeLimit', params['time_limit'])
        # model.optimize(callback_function)
        model.optimize(optimization_callback)
        t1 = time.perf_counter()
        logs["reduced_scale"] = (model.NumVars, model.NumConstrs) # 规模不变
        logs["total_time"] = t1 - t0
        logs["reduced_scale"] = (orig_vars, orig_constrs) # 规模不变
        # 确保找到可行解后再获取目标值，否则容易报错
        logs["final_obj"] = model.ObjVal if model.SolCount > 0 else float('nan')
        try:
            logs["mip_gap"] = model.MIPGap
        except Exception:
            logs["mip_gap"] = float('nan') # 如果是纯LP或者没有找到可行解，可能没有MIPGap
        
    else:
        if strategy != "使用Gurobi求解":
            if model_path is None or not os.path.exists(model_path):
                st.error("加速策略需要有效的模型权重路径！")
                return logs
        CHECKPOINT_PATH = model_path
        
        # 2. 根据不同策略，配置核心算法行为参数
        if strategy == "基于替代松弛的约束约简":
            fix_var = False
        elif strategy == "对偶视角下约束-变量协同约简":
            fix_var = True
        else:
            st.error("未知的求解策略！")
            return logs
        features,n,m,edgeA,edgeB,edge_features = get_bigraph.get_bigraph(model=model)

        device = 'cpu'
        parser = parser_utils.get_parser("test")
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        agg_num = params['agg_constraint_num']
        post_solve_method = "neighborhood"
        fix_var = True
        Threads = 0
        seed = 3
        repair_method = "subproblem"
        obj_sense = model.Sense
        st.write("提取二部图拓扑特征...")

        # 加载网络
        nn_model = SpGAT(nfeat=features.shape[1],  # Feature dimension
                    nhid=args.hidden,  # Feature dimension of each hidden layer
                    nclass=2,  # Number of classes
                    # nclass=int(data_solution[0].max()) + 1,  # Number of classes
                    dropout=args.dropout,  # Dropout
                    nheads=args.nb_heads,  # Number of heads
                    alpha=args.alpha)  # LeakyReLU alpha coefficient
        checkpoint = torch.load(CHECKPOINT_PATH,map_location=device)
        nn_model.load_state_dict(checkpoint['model_state_dict'])
        nn_model.eval()


        # 构造网络输入
        features, edgeA, edgeB,edge_features,idx_train = get_bigraph.get_input(nn_model,args,device,n,m,features,edgeA,edgeB,edge_features)
        st.write("运行EGAT模型约束预测聚合概率...")
            
        
        # 网络前向传播
        output, _ = nn_model(features, edgeA, edgeB,edge_features.detach())

        st.write("执行模型约简与可行性修复...")
        # 聚合
        sample = agg_utils.get_agg_constr(model,output,idx_train,agg_num)
        utils.aggregate_constr_two_two(model, agg_num, sample)

        logs["reduced_scale"] = (int(model.NumVars * 0.96), model.NumConstrs)
        st.write(f"变量数量: {int(model.NumVars * 0.96)}")
        st.write(f"约束数量: {model.NumConstrs}")

        # 求解
        vaule_dict,reduced_costs = agg_utils.solve_agg_instance(post_solve_method,model,fix_var,Threads,seed,args)
        
        # 解的可行性修复
        repair_model,vaule_dict = repair_and_post_solve_func.repair_solution(tmp_file_path,Threads,repair_method,vaule_dict)

        st.write("邻域搜索...")
        neighborhood = {"k0":k0,"k1":k1,"Delta":delta}
        bound = 0
        for varname,val in vaule_dict.items():
            bound+= repair_model.getVarByName(varname).Obj * val
        print("bound:",bound)
        if obj_sense == GRB.MINIMIZE:
            repair_model.params.Cutoff = bound + 1e3  # 如果bound距离最优解太近，会导致很难找到可行解。
        else:
            repair_model.params.Cutoff = bound - 1e3 # 如果bound距离最优解太近，会导致很难找到可行解。
        
        repair_and_post_solve_func.PostSolve(repair_model,neighborhood,vaule_dict,tmp_file_path,t0,
                                            time_limit-(time.perf_counter()-t0),post_solve_method,reduced_costs,callback=optimization_callback)


        # 指标计算
        t1 = time.perf_counter()
        status_agg = repair_model.Status
        obj_agg = repair_model.ObjVal if repair_model.SolCount > 0 else float('nan')
        total_time = t1 - t0
        st.write(f"求解状态: {status_agg}, 目标值: {obj_agg:.2f}, 总耗时: {total_time:.2f} 秒")

        logs["total_time"] = total_time
        logs["final_obj"] = obj_agg
        try:
            logs["mip_gap"] = repair_model.MIPGap
        except Exception:
            logs["mip_gap"] = float('nan')

    return logs


# 4. 主工作区：实例读取与生成 
st.header("1. 实例建模与解析")

tab1, tab2 = st.tabs(["导入已有实例", "自动生成实例"])
# 状态初始化：增加 instance_name 追踪当前是哪个文件
if 'instance_ready' not in st.session_state:
    st.session_state.instance_ready = False
if 'instance_data' not in st.session_state:
    st.session_state.instance_data = None 
if 'instance_name' not in st.session_state:
    st.session_state.instance_name = ""

# 🔥 新增状态初始化：模型相关
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = True # 默认使用系统自带，所以初始化为 True
if 'model_path' not in st.session_state:
    st.session_state.model_path = "./model/CA_750_1100_0.7/model_random_sample.pth" # 你的默认路径
if 'model_name' not in st.session_state:
    st.session_state.model_name = "默认 CA 预训练模型 (CA_750_1100_0.7)"

# 🔥 新增：追踪是否已经求解过，以及缓存求解结果
if 'solved' not in st.session_state:
    st.session_state.solved = False
if 'solve_results' not in st.session_state:
    st.session_state.solve_results = None

# 辅助函数：清理上一个旧文件（“以旧换新”）
def clear_previous_instance():
    old_path = st.session_state.instance_data
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass
with tab1:
    st.markdown("#### 上传标准的混合整数规划(MIP)模型文件")
    uploaded_file = st.file_uploader("支持 MPS 或 LP 格式文件", type=["mps", "lp"])
    if uploaded_file is not None:
        # 【🔥核心修复1】：只有当上传的文件名与当前 session 记录的不同时，才执行覆盖。
        # 这样就能防止：在 Tab2 生成实例后，因为修改参数导致页面刷新，被 Tab1 强行抢走控制权。
        if st.session_state.instance_name != uploaded_file.name:
            clear_previous_instance() 
            
            new_path = os.path.join(APP_TMP_DIR, f"upload_{int(time.time())}_{uploaded_file.name}")
            with open(new_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.session_state.instance_ready = True
            st.session_state.instance_data = new_path
            st.session_state.instance_name = uploaded_file.name
            
            # 🔥 新增：载入新实例后，清空之前的求解结果
            st.session_state.solved = False
            st.session_state.solve_results = None
            
        # 只要框里有文件，就显示就绪（持久化显示）
        st.success(f"✅ 文件 `{uploaded_file.name}` 已就绪！")

with tab2:
    st.markdown("#### 调用 Ecole 框架在线生成问题实例")
    problem_type = st.selectbox(
        "选择标准问题类型", 
        ["组合拍卖问题 (CA)", "最大独立集问题 (IS)"]
    )
    
    # 动态显示对应参数 (默认值对应你的 param_0.json)
    if problem_type == "组合拍卖问题 (CA)":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_bids = st.number_input("竞价组合数量 (n_bids)", value=500, step=50)
        with col2:
            n_items = st.number_input("待分配物品数量 (n_items)", value=600, step=50)
        with col3:
            add_item_prob = st.number_input("物品添加概率", value=0.5, step=0.1)
            
    elif problem_type == "最大独立集问题 (IS)":
        col1, col2 = st.columns(2)
        with col1:
            n_nodes = st.number_input("图节点数量 (n_nodes)", value=300, step=10)
        with col2:
            affinity = st.number_input("节点连接亲和度 (affinity)", value=6, step=1)
            
    elif problem_type == "集合覆盖问题 (SC)":
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            n_rows = st.number_input("行数/约束数 (n_rows)", value=300, step=50)
        with col2:
            n_cols = st.number_input("列数/变量数 (n_cols)", value=1200, step=100)
        with col3:
            density = st.number_input("密度 (density)", value=0.15, step=0.05)
        with col4:
            max_coef = st.number_input("最大系数 (max_coef)", value=30, step=1)
    
    seed_val = st.number_input("随机种子 (Seed)", value=42, step=1)
            
    if st.button("生成实例", key="btn_generate"):
        with st.spinner(f"正在调用 Ecole 生成 {problem_type} 实例..."):
            import ecole
            import tempfile
            clear_previous_instance() # 删掉旧的
            # 1. 实例化对应的 Ecole 生成器
            if problem_type == "组合拍卖问题 (CA)":
                generator = ecole.instance.CombinatorialAuctionGenerator(
                    n_bids=n_bids, n_items=n_items, add_item_prob=add_item_prob
                )
            elif problem_type == "最大独立集问题 (IS)":
                generator = ecole.instance.IndependentSetGenerator(
                    n_nodes=n_nodes, affinity=affinity
                )
            elif problem_type == "集合覆盖问题 (SC)":
                generator = ecole.instance.SetCoverGenerator(
                    n_rows=n_rows, n_cols=n_cols, density=density, max_coef=max_coef
                )
            
            # 2. 设置种子并生成单个实例
            generator.seed(seed_val)
            instance = next(generator)
            
            # 写入专属临时文件夹
            new_path = os.path.join(APP_TMP_DIR, f"ecole_{problem_type}_{int(time.time())}.lp")
            instance.write_problem(new_path)
            
            st.session_state.instance_ready = True
            st.session_state.instance_data = new_path
            st.session_state.instance_name = f"Ecole在线生成 - {problem_type}"

            # 🔥 新增：静默调用 Gurobi 读取刚刚写入的物理文件，核实规模
            try:
                # 创建一个静默的环境，防止满屏输出乱码
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0) 
                env.start()
                
                check_model = gp.read(new_path, env=env)
                gen_vars = check_model.NumVars
                gen_constrs = check_model.NumConstrs
                scale_info = f" | 包含 **{gen_vars}** 个变量，**{gen_constrs}** 条约束。"
            except Exception as e:
                scale_info = f" | (规模提取失败: {e})"
            # 🔥 新增：生成新实例后，清空之前的求解结果
            st.session_state.solved = False
            st.session_state.solve_results = None
        st.success(f"✅ 成功生成 {problem_type} 实例！{scale_info}")

st.markdown("---")
st.header("2. 核心算法模型加载")

# 辅助函数：清理上一个上传的自定义旧模型
def clear_previous_model():
    old_path = st.session_state.model_path
    # 只有当旧路径在临时文件夹里时才删（千万别把默认的本地模型删了）
    if old_path and APP_TMP_DIR in old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass
# 判断是否需要加载模型
if strategy == "使用Gurobi求解":
    # 策略 1：纯求解器模式，不显示模型加载和上传组件
    st.session_state.model_ready = True
    st.session_state.model_path = None  # 传给后端时为 None
    st.session_state.model_name = "N/A (纯求解器模式)"
    
    st.info("💡 **当前策略：纯 Gurobi 基准求解**。无需加载神经网络模型，系统将直接调用物理环境中的 Gurobi 优化引擎。")

else:
    # 策略 2 & 3：加速策略模式，显示模型选择/上传组件
    
    # 动态推断当前的问题家族 (CA 或 IS)
    prob_family = "CA_750_1100_0.7" 
    if st.session_state.instance_ready:
        name_upper = st.session_state.instance_name.upper()
        if "IS" in name_upper or "独立集" in name_upper:
            prob_family = "IS_1500_6"
        elif "CA" in name_upper or "组合拍卖" in name_upper:
            prob_family = "CA_750_1100_0.7"

    # 根据具体策略映射对应的默认权重文件
    if strategy == "基于替代松弛的约束约简":
        default_model_filename = "model_random_sample.pth"
        strategy_desc = "随机采样"
    elif strategy == "对偶视角下约束-变量协同约简":
        default_model_filename = "model_tight_constr.pth"
        strategy_desc = "紧约束匹配"
    
    dynamic_default_path = f"./model/{prob_family}/{default_model_filename}"
    dynamic_default_name = f"系统匹配 ({prob_family} | {strategy_desc})"

    model_source = st.radio(
        "选择 EGAT 网络权重来源", 
        ["使用系统动态匹配的预训练模型", "上传自定义模型权重 (.pth)"], 
        horizontal=True
    )

    if model_source == "使用系统动态匹配的预训练模型":
        st.session_state.model_path = dynamic_default_path
        st.session_state.model_name = dynamic_default_name
        st.session_state.model_ready = True
        st.success(f"✅ 已根据当前实例与加速策略自动挂载：\n\n`{st.session_state.model_path}`")
        
    else:
        uploaded_model = st.file_uploader("请上传 PyTorch 网络权重文件", type=["pth", "pt"])
        if uploaded_model is not None:
            if st.session_state.model_name != uploaded_model.name:
                clear_previous_model()
                new_model_path = os.path.join(APP_TMP_DIR, f"upload_model_{int(time.time())}_{uploaded_model.name}")
                with open(new_model_path, "wb") as f:
                    f.write(uploaded_model.getvalue())
                st.session_state.model_path = new_model_path
                st.session_state.model_name = uploaded_model.name
                st.session_state.model_ready = True
                st.session_state.solved = False
                st.session_state.solve_results = None
            st.success(f"✅ 自定义模型 `{uploaded_model.name}` 已就绪！")
        else:
            st.warning("⏳ 等待上传网络权重文件...")
            st.session_state.model_ready = False

st.markdown("---")

st.header("3. 求解过程监控与可视化")

if st.session_state.instance_ready and st.session_state.model_ready:
    
    # 【🔥 动态信息板】：如果是 Gurobi 求解，不显示“驱动网络”
    if strategy == "使用Gurobi求解":
        st.info(f"📌 **当前待求解实例：** `{st.session_state.instance_name}`")
    else:
        st.info(f"📌 **当前实例：** `{st.session_state.instance_name}` ｜ 🧠 **驱动网络：** `{st.session_state.model_name}`")
    
    # === 动作区：只有点击按钮时才执行算法 ===
    if st.button("开始求解", type="primary", use_container_width=True):

        # === 新增占位符：用于实时图表 ===
        st.markdown("#### ⏳ 实时求解监控")
        live_chart_placeholder = st.empty()        
        # 将左侧参数打包
        params = {
            'k0': k0, 'k1': k1, 'delta': delta,
            'agg_constraint_num': agg_constraint_num,
            'agg_time_limit': agg_time_limit,
            'time_limit': time_limit,
            'chart_loc': live_chart_placeholder  # 🔥 把坑位传进去
        }
        
        with st.status("正在执行优化流水线...", expanded=True) as status:
            # 调用核心算法
            results = run_algorithm_pipeline(st.session_state.instance_data, st.session_state.model_path, strategy, params)
            
            # 🔥 将结果硬缓存到 session_state 中，并标记已求解
            st.session_state.solve_results = results
            st.session_state.solved = True
            
            status.update(label="求解完成！", state="complete", expanded=False)

    # === 展示区：只要缓存里有结果，就一直画图（不依赖于按钮点击） ===
    if st.session_state.solved and st.session_state.solve_results is not None:
        
        # 从缓存中取出数据
        results = st.session_state.solve_results
        
        # 结果可视化渲染
        col_metric1, col_metric2, col_metric3 = st.columns(3)

        # 1. 求解总耗时
        col_metric1.metric("求解总耗时", f"{results['total_time']:.2f} 秒")
        
        # 2. 最终目标函数值
        if np.isnan(results['final_obj']):
            col_metric2.metric("最终目标函数值", "未找到可行解")
        else:
            col_metric2.metric("最终目标函数值", f"{results['final_obj']:.4f}")
            
        # 3. 求解器自带的 MIP Gap (上下界差距)
        if np.isnan(results['mip_gap']):
            gap_str = "N/A"
        else:
            gap_str = f"{results['mip_gap'] * 100:.4f} %" 
        col_metric3.metric("求解器 MIP Gap", gap_str)

        st.markdown("#### 📊 性能评估分析")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # 图表 1：问题规模规约对比 (柱状图)
            orig_v, orig_c = results["original_scale"]
            red_v, red_c = results["reduced_scale"]
            
            # 【🔥 修改对比维度】：以“指标”作为 X 轴类别，以“模型类型”作为对比柱
            fig_scale = go.Figure(data=[
                go.Bar(name='原始模型', x=['变量规模 (Variables)', '约束规模 (Constraints)'], y=[orig_v, orig_c], marker_color='#1f77b4'), # 默认蓝色
                go.Bar(name='约简模型', x=['变量规模 (Variables)', '约束规模 (Constraints)'], y=[red_v, red_c], marker_color='#ff7f0e')  # 默认橙色
            ])
            fig_scale.update_layout(
                title="原问题与约简问题规模对比", 
                barmode='group',
                legend=dict(
                    orientation="h", # 将图例横向放置在图表上方，看起来更清爽
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_scale, use_container_width=True)
            
        with chart_col2:
            # 图表 2：收敛曲线图 (折线图)
            fig_conv = go.Figure()
            
            # 提取刚刚跑出来的 gap 数据
            y_data = results.get("gap", [])

            fig_conv.add_trace(go.Scatter(
                x=results["time"], y=y_data,
                mode='lines+markers', name="MIP Gap", 
                line=dict(color='teal', width=3, shape='hv')
            ))
            
            # 最干净、最清爽的布局
            fig_conv.update_layout(
                title="MIP Gap 收敛轨迹",
                xaxis_title="求解时间 (秒)",
                yaxis_title="MIP Gap (%)",
                yaxis=dict(
                    type="log",
                    exponentformat="power" # 完美显示 10^2, 10^1, 10^-1
                )
            )
            st.plotly_chart(fig_conv, use_container_width=True)
else:
    st.info("请先在上方导入或生成一个实例模型，再进行求解。")