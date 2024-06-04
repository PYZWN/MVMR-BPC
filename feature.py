# import pickle
# import numpy as np
# import os

# # 输入文件路径
# input_file_path = '/hy-tmp/newfeature.pkl'
# # 输出文件路径
# output_file_path = '/hy-tmp/sequencefeature.pkl'

# # 读取包含所有token_representations的pkl文件
# with open(input_file_path, 'rb') as file:
#     all_features = pickle.load(file)

# # 准备一个字典来存储所有的sequence_representations
# all_sequence_representations = {}

# # 处理每个序列的特征
# for label, token_representations in all_features.items():
#     # 计算sequence_representation
#     if token_representations.shape[0] > 2:  # 检查序列长度是否足够
#         sequence_representation = token_representations[1:-1].mean(axis=0)
#     else:
#         # 如果序列长度不足以进行平均，则直接使用整个表示
#         sequence_representation = token_representations.mean(axis=0)

#     # 将计算的平均特征保存到字典中
#     all_sequence_representations[label] = sequence_representation

# # 将所有的sequence_representations保存到一个pkl文件
# with open(output_file_path, 'wb') as f:
#     pickle.dump(all_sequence_representations, f)

# print("All sequence features have been processed and saved in one pkl file.")


import pickle

# 输入文件路径
input_file_path = 'model_data.pkl'

# 读取pkl文件
with open(input_file_path, 'rb') as file:
    data = pickle.load(file)

# 打印数据的数量
print("Number of entries in the dataset:", len(data))

# 打印每个数据的形状（假设数据是numpy数组）
for label, array in list(data.items())[:10]:  # 取前10个数据
    print(f"Label: {label}, Shape: {array.shape}")

# 打印前10个数据的具体内容
for label, array in list(data.items())[:10]:
    print(f"Label: {label}, Data:\n{array}\n")

# import pickle

# # 文件路径
# file_path = './net_data/net_file_60.pkl'

# # 要删除的键
# key_to_remove = 'GICACRRRFCPNSERFSGYCRVNGARYVRCCSRR'

# # 读取pkl文件
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # 检查键是否在字典中，并删除
# if key_to_remove in data:
#     print(f"Key '{key_to_remove}' found in the data. Deleting...")
#     del data[key_to_remove]
#     # 保存修改后的数据到原文件
#     with open(file_path, 'wb') as file:
#         pickle.dump(data, file)
#     print("The key has been deleted and the file has been updated.")
# else:
#     print(f"Key '{key_to_remove}' not found in the data. No deletion performed.")


