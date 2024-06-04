import numpy as np
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_sequences(file_paths):
    sequences = []
    labels = []
    for file_path in file_paths:
        label = 'AHP' if 'AHP' in file_path else 'AIP'
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(1, len(lines), 2):
                sequence = lines[i].strip()
                labels.append(label)
                sequences.append(sequence)
    return labels, sequences

def sequence_to_vector(sequences, maxlen=517):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sequences)
    sequences_encoded = tokenizer.texts_to_sequences(sequences)
    sequences_padded = pad_sequences(sequences_encoded, maxlen=maxlen, padding='post')
    return sequences_padded

def plot_tsne(data, labels, perplexity=30, save_path_png='tsne_plot.png', save_path_eps='tsne_plot.eps'):
    tsne = TSNE(n_components=2, n_iter=3000, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(data)
    
    plt.figure(figsize=(16, 10))
    unique_labels = list(set(labels))
    colors = {'AHP': 'purple', 'AIP': 'orange'}  # 修改颜色
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, color=colors[label], alpha=0.8)  # 设置透明度
    
    plt.legend(fontsize=30)  # 修改图例的大小
    plt.title("AHP and AIP", fontsize=40)  # 修改标题的大小
    plt.xlabel("T-SNE Feature 1", fontsize=30)  # 修改X轴标签的大小
    plt.ylabel("T-SNE Feature 2", fontsize=30)  # 修改Y轴标签的大小
    plt.xticks(fontsize=30)  # 修改X轴刻度的大小
    plt.yticks(fontsize=30)  # 修改Y轴刻度的大小
    plt.savefig(save_path_png)
    plt.savefig(save_path_eps)
    plt.show()

# 输入TXT文件的路径
file_paths = ['data1/AHP/AHPCD_.txt', 'data1/AIP/AIPCD_.txt']

# 读取序列和标签
labels, sequences = read_sequences(file_paths)

# 将序列转换为向量
sequences_vector = sequence_to_vector(sequences)

# 进行T-SNE聚类分析并可视化，并保存图片
plot_tsne(sequences_vector, labels, perplexity=30, save_path_png='AHPAIP.png', save_path_eps='AHPAIP.eps')
