# test_random_cons.py
用来测试随机聚合，以及各种修复函数的效果

# gen_train_data.py
生成训练用的二部图、标签

# EGAT_layers.py
Define the layers of the EGAT model.

# EGAT_models.py:
Define the overall EGAT model architecture.

# train.py
Train the EGAT model using the generated datasets.

# test.py
Run the trained model on test datasets to obtain optimized results.

# gen_assignment_problem.py
用来生成指派问题的脚本，基本用不上

# gen_general_CA.py
用来生成含有小于等于、大于等于约束的CA问题，这是为了测试可行性修复是否能够适应于大于等于约束

# generateInstance.py
用来生成CA、IS、SC等问题

# gurobi.py
用不上了，之前ps框架里用来生成二部图和训练集的

# helper.py
同上，用不上

# main.py
用不上

# parser_utils.py
用来读取和网络、求解相关的参数，单独写在一个函数里，是为了保证训练和测试是一致的

# repair_and_post_solve_func.py
用于修复、邻域搜索的函数

# utils.py
存放了一些常用的函数