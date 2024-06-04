import os
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras import utils
import pickle
from evaluation import scores, evaluate
from mamba import MambaBlock,ResidualBlock,Mamba,RMSNorm
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Conv1D, GlobalAveragePooling1D, LSTM, Flatten, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import numpy as np
np.random.seed(101)
from pathlib import Path



def catch(data, data2, data3, data4, data5, data6,label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            print(t)
            chongfu += 1
            print(data[t[0]])
            print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        data2 = np.delete(data2, idx, axis=0)
        data3 = np.delete(data3, idx, axis=0)
        data4 = np.delete(data4, idx, axis=0)
        data5 = np.delete(data5, idx, axis=0)
        data6 = np.delete(data6, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, data2, data3, data4, data5, data6,label



from model import MKey_Net_DiladCNNBiLSTM_Attention, MKey_Net_DiladCNNBiLSTM_GCN_Attention, MKey_Net_DiladCNNBiLSTM_GCN_GCN_AttentionCat,TLULayer,FRNLayer


import numpy as np

def compute_shap_values_batch(explainer, data, batch_size):
    # 分割数据为批次
    num_samples = len(data[0])
    shap_values = [None] * num_samples  # 初始化一个列表来存储SHAP值

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_data = [d[start:end] for d in data]  # 提取当前批次的数据
        batch_shap_values = explainer.shap_values(batch_data)  # 计算当前批次的SHAP值
        # 存储SHAP值
        for i, sv in enumerate(batch_shap_values):
            if shap_values[i] is None:
                shap_values[i] = sv
            else:
                shap_values[i] = np.vstack((shap_values[i], sv))
    
    return shap_values




def train_my(train, test, para, model_num, model_path):

    Path(model_path).mkdir(exist_ok=True)

    # data get
    X_train, K_train, N_train, F_train, A_train, E_train, y_train = train[0], train[1], train[2], train[3], train[4], train[5], train[6]

    # data and label preprocessing
    y_train = tf.keras.utils.to_categorical(y_train)
    X_train, K_train, N_train, F_train, A_train, E_train, y_train = catch(X_train, K_train, N_train, F_train, A_train, E_train, y_train)
    y_train[y_train > 1] = 1

    # disorganize
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    K_train = K_train[index]
    N_train = N_train[index]
    F_train = F_train[index]
    A_train = A_train[index]
    E_train = E_train[index]
    y_train = y_train[index]

    # train
    length = X_train.shape[1]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon,t_data.tm_mday,t_data.tm_hour,t_data.tm_min,t_data.tm_sec))

    # test数据处理
    test[6] = utils.to_categorical(test[6])
    test[0], test[1], test[2], test[3], test[4], test[5], temp = catch(test[0], test[1], test[2], test[3], test[4], test[5], test[6])
    temp[temp > 1] = 1
    test[6] = temp
    y_test = temp
    thred = 0.5


    for counter in range(model_num):
        if model_path == 'MKey_Net_DiladCNNBiLSTM_GCN_Attention':
            model = MKey_Net_DiladCNNBiLSTM_GCN_Attention(length, out_length, para, feature_vis=True)

        # 训练模型
        model.fit([X_train, K_train, N_train, F_train, A_train, E_train], y_train, epochs=50, batch_size=32, verbose=2)
        
        # 特征提取用于T-SNE可视化
        features, _ = model.predict([test[0], test[1], test[2], test[3], test[4], test[5]])
        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000)
        tsne_results = tsne.fit_transform(features)
        original_labels = np.argmax(test[6], axis=1)  # 从独热编码转换为类别索引

# 将特征和标签存储在同一个字典中
        data_to_save = {
            'features': features,
            'labels': original_labels
        }
        with open('model_data.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)

        label_colors = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'purple',
            4: 'orange'
        }

# 创建颜色数组
        colors = [label_colors[label] for label in original_labels]

# 定义标记
        markers = ['o', 'o', 'o', 'o', 'o']  # 圆圈，方块，三角，五角，星形

# 定义新的图例标签
        class_labels = {
            0: 'AMP',
            1: 'ACP',
            2: 'ADP',
            3: 'AHP',
            4: 'AIP'
        }

        plt.figure(figsize=(15, 15))
        for i, marker in zip(label_colors.keys(), markers):
            plt.scatter(tsne_results[original_labels == i, 0], tsne_results[original_labels == i, 1], 
                        c=label_colors[i], marker=marker, label=class_labels[i], alpha=0.6)

        plt.title('T-SNE Visualization', fontsize=30)
        plt.xlabel('Component 1', fontsize=30)
        plt.ylabel('Component 2', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

# 创建自定义图例
        handles = []
        for label, color in label_colors.items():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=class_labels[label]))

        plt.legend(handles=handles, title="Classes", fontsize=25, title_fontsize=25)  # 增大图例字体大小

# 保存图像为 PNG 和 EPS 格式
        plt.savefig('enhanced_tsne_visualization.png')
        plt.savefig('enhanced_tsne_visualization.eps', format='eps')

        plt.show()
        plt.close()

# 输出路径信息，确认保存位置
        print("Visualization saved to enhanced_tsne_visualization.png and enhanced_tsne_visualization.eps")
        # 预测结果
        score, features = model.predict([test[0], test[1], test[2], test[3], test[4], test[5]])
        # 在模型预测后评估模型性能
        thred = 0.5  # 可以调整这个阈值根据具体情况
        binary_predictions = (score > thred).astype(int)
        y_test_indices = np.argmax(test[6], axis=1)
        binary_predictions_indices = np.argmax(binary_predictions, axis=1)
        accuracy = accuracy_score(y_test_indices, binary_predictions_indices)
        print(f"Overall Accuracy: {accuracy}")
        report = classification_report(y_test_indices, binary_predictions_indices)
        print("Classification Report:")
        print(report)
        conf_matrix = confusion_matrix(y_test_indices, binary_predictions_indices)
        print("Confusion Matrix:")
        print(conf_matrix)
        # 可视化混淆矩阵
        # 可视化混淆矩阵
        plt.figure(figsize=(15, 15))  # 增加图像尺寸
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['AMP', 'ACP', 'ADP', 'AHP', 'AIP'], 
                    yticklabels=['AMP', 'ACP', 'ADP', 'AHP', 'AIP'], 
                    annot_kws={"size": 30})  # 增大框内数字字体大小
        plt.xlabel('Predicted', fontsize=30)  # 增大坐标轴上的标签字体大小
        plt.ylabel('True', fontsize=30)  # 增大坐标轴上的标签字体大小
        plt.title('MVMR-BPC Confusion Matrix Heatmap', fontsize=30)  # 增大标题字体大小
        plt.xticks(fontsize=25)  # 增大坐标轴上的数字字体大小
        plt.yticks(fontsize=25)  # 增大坐标轴上的数字字体大小

        # 保存图像为 PNG 和 EPS 格式
        plt.savefig('GMBP_confusion_matrix_heatmap.png')
        plt.savefig('GMBP_confusion_matrix_heatmap.eps', format='eps')

        plt.show()
        plt.close()

# 输出路径信息，确认保存位置
        print("Visualization saved to GMBP_confusion_matrix_heatmap.png and GMBP_confusion_matrix_heatmap.eps")
        "========================================"
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] < thred:
                    score[i][j] = 0
                else:
                    score[i][j] = 1
        a, b, c, d, e = evaluate(score, y_test)
        print(a, b, c, d, e)
        "========================================"

        # 3.evaluation
        if counter == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / model_num

    # data saving
    with open(os.path.join(model_path, 'MLBP_prediction_prob.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred: score_label[i][j] = 0
            else: score_label[i][j] = 1

    # data saving
    with open(os.path.join(model_path, 'MLBP_prediction_label.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    out = model_path
    Path(out).mkdir(exist_ok=True, parents=True)
    out_path2 = os.path.join(out, 'result_test.txt')
    with open(out_path2, 'w') as fout:
        fout.write('aiming:{}\n'.format(aiming))
        fout.write('coverage:{}\n'.format(coverage))
        fout.write('accuracy:{}\n'.format(accuracy))
        fout.write('absolute_true:{}\n'.format(absolute_true))
        fout.write('absolute_false:{}\n'.format(absolute_false))
        fout.write('\n')

        



import time
from test import test_my
def train_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    train_my(train, test, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test start time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

    # test_my(test, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

def test_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}



    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test start time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

    test_my(test, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

