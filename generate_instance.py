import os
import time
from multiprocessing import Process, Queue
# import networkx as nx
import random
from pyscipopt import Model as ScipModel
import ecole
import os
import utils

os.environ["LD_DEBUG"] = "libs"

# prefix = '../../../../project/predict_and_search/'
prefix = './'


# prefix  = ''
# import networkx as nx
import ecole.scip


class MVCGenerator:
    """自定义的最小顶点覆盖(MVC)生成器"""
    def __init__(self, num_nodes=6000, target_edges=None, barabasi_m=None):
        self.num_nodes = num_nodes
        self.target_edges = target_edges

        if target_edges is not None:
            estimated_m = max(1, int(target_edges / num_nodes))
            self.barabasi_m = estimated_m
        else:
            self.barabasi_m = barabasi_m if barabasi_m is not None else 5

        self._base_seed = None

    def seed(self, seed):
        self._base_seed = seed
        random.seed(seed)

    def __next__(self):
        if self._base_seed is None:
            raise ValueError("请先调用 generator.seed(...) 设置随机种子。")

            # 生成BA图
        G = nx.barabasi_albert_graph(
            n=self.num_nodes,
            m=self.barabasi_m,
            seed=self._base_seed
        )
        self._base_seed += 1

        # 如果指定了目标边数且当前边数过多，随机删除一些边
        current_edges = G.number_of_edges()
        if self.target_edges is not None and current_edges > self.target_edges:
            edges_to_remove = random.sample(list(G.edges()), current_edges - self.target_edges)
            G.remove_edges_from(edges_to_remove)

            # 打印实际边数（调试用）
        print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

        # 使用PySCIPOpt创建模型
        model = ScipModel("mvc")

        # 添加变量
        x_vars = {}
        for node in G.nodes():
            x_vars[node] = model.addVar(f"x_{node}", vtype="B")

            # 设置目标函数
        model.setObjective(sum(x_vars.values()), "minimize")

        # 添加约束
        for u, v in G.edges():
            model.addCons(x_vars[u] + x_vars[v] >= 1)

            # 保存模型到临时文件
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.lp', delete=False)
        model.writeProblem(temp_file.name)
        temp_file.close()

        # 使用Ecole加载模型
        ecole_model = ecole.scip.Model.from_file(temp_file.name)

        # 删除临时文件
        import os
        os.unlink(temp_file.name)

        return ecole_model

    def __iter__(self):
        return self

def generate_single_instance(n, queue, dataset_name, problem_type, generator, seed, post_fix):
    generator.seed(seed)
    while True:
        i = queue.get()
        if i is None:
            break  # No more tasks

        instance = next(generator)
        instance_dir = prefix + f"instance/{dataset_name}/{problem_type}_{post_fix}"
        os.makedirs(instance_dir, exist_ok=True)
        instance_path = os.path.join(instance_dir, f"{problem_type}_{n + i}.lp")
        instance.write_problem(instance_path)
        print(f"第{n + i}个问题实例已生成：{instance_path}")


def get_configured_generator(problem_type: str,problem_parameters):
    """
    根据问题类型键，从 problem_parameters 字典中获取参数，
    并实例化对应的 Ecole 生成器。
    """
    if problem_type not in problem_parameters:
        raise ValueError(f"未知的问题类型: {problem_type}")

    # 获取该问题类型对应的参数字典
    params = problem_parameters[problem_type]
    # 根据问题类型实例化不同的生成器
    if problem_type == "CA":
        # 500, 600, add_item_prob=0.7普遍是几百到一千秒
        generator = ecole.instance.CombinatorialAuctionGenerator(**params)
    elif problem_type == "IS":
        generator = ecole.instance.IndependentSetGenerator(**params)
    else:
        raise NotImplementedError(f"生成器类型 '{problem_type}' 尚未实现。")

    return generator


def generate_instances(num_instances, dataset_name, problem_type,num_workers,base_seed,problem_parameters):
    # if size == "CF":
    #     generator = ecole.instance.CapacitatedFacilityLocationGenerator(50, 100)
    # elif size == "IS": # 4000
    #     # generator = ecole.instance.IndependentSetGenerator(6000)
    #     generator = ecole.instance.IndependentSetGenerator(600)
    # elif size == "IS_hard": # 6000
    #     generator = ecole.instance.IndependentSetGenerator(9000)
    # elif size == "CA":
    #     if epoch == 0:
    #         # generator = ecole.instance.CombinatorialAuctionGenerator(300, 1500)
    #         generator = ecole.instance.CombinatorialAuctionGenerator(30, 150)
    #     elif epoch == 1:
    #         # generator = ecole.instance.CombinatorialAuctionGenerator(2000, 4000)
    #         # generator = ecole.instance.CombinatorialAuctionGenerator(100, 200)
    #         # generator = ecole.instance.CombinatorialAuctionGenerator(500, 700)
    #         # generator = ecole.instance.CombinatorialAuctionGenerator(7000, 1500)
    #         generator = ecole.instance.CombinatorialAuctionGenerator(8600, 1500,add_item_prob=0.85) # 差不多达到论文的标准
    #         # generator = ecole.instance.CombinatorialAuctionGenerator(500, 1500)
    # elif size == "CA_hard":
    #     if epoch == 0:
    #         generator = ecole.instance.CombinatorialAuctionGenerator(600, 3000)
    #     elif epoch == 1:
    #         generator = ecole.instance.CombinatorialAuctionGenerator(3000, 6000)
    # elif size == "SC":
    #     generator = ecole.instance.SetCoverGenerator(3000, 5000)
    # elif size == "MVC":
    #     # MVC: med，默认 barabasi_m=5
    #     generator = MVCGenerator(num_nodes=4000)
    # elif size == "MVC_hard":
    #     # MVC_hard: 6000 节点
    #     generator = MVCGenerator(num_nodes=6000)
    # else:
    #     raise ValueError("Invalid type")


    # generator = generators_dict[problem_type]
    generator = get_configured_generator(problem_type,problem_parameters)
    param = problem_parameters[problem_type]
    post_fix = utils.get_post_fix(param)
    # observation_function = ecole.observation.MilpBipartite()


    # Create a queue to hold tasks
    task_queue = Queue()
    n = 0
    for i in range(num_instances):
        task_queue.put(i)

    # Create worker processes
    workers = []
    for worker_id in range(num_workers):
        seed = base_seed + worker_id # 每个worker要有不同的seed
        worker = Process(target=generate_single_instance,
                         args=(n, task_queue, dataset_name, problem_type, generator, seed, post_fix))
        workers.append(worker)
        worker.start()

    # Add None to the queue to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    num_workers = 20

    # 读取生成实例的参数
    json_file_path = 'parameters/param_3.json'
    problem_parameters = utils.get_problem_parameters(json_file_path)

    # ecole.instance.CombinatorialAuctionGenerator(2000, 1000, add_item_prob=0.8), # 3600解不完，10%gap左右
    # dataset_name = "train"
    dataset_name = "test"
    if dataset_name == "train":
        base_seed = 47
    else:
        base_seed = 114514
    generate_instances(10, dataset_name, "CA",num_workers,base_seed,problem_parameters)
    generate_instances(10, dataset_name, "IS",num_workers,base_seed,problem_parameters)


