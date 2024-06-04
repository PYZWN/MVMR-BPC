# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# import graphviz
# import pickle
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.tree import export_graphviz
# from evaluation import scores, evaluate
# from sklearn.preprocessing import LabelBinarizer
# # 假设特征和标签已从文件加载
# with open('model_data.pkl', 'rb') as f:
#     model_data = pickle.load(f)

# features = model_data['features']
# labels = model_data['labels']

# # 训练决策树
# tree = DecisionTreeClassifier(max_depth=5)
# tree.fit(features, labels)
# # 使用决策树模型进行预测
# tree_predictions = tree.predict(features)

# # 如果labels为多分类标签，需要转换为二进制形式
# lb = LabelBinarizer()
# labels_bin = lb.fit_transform(labels)
# tree_predictions_bin = lb.transform(tree_predictions)  # 转换预测结果为二进制标签形式


# # 计算性能指标
# accuracy = accuracy_score(labels, tree_predictions)
# class_report = classification_report(labels, tree_predictions)
# conf_matrix = confusion_matrix(labels, tree_predictions)

# # 使用您定义的评价函数
# aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(tree_predictions_bin, labels_bin)

# # 打印您定义的评价指标结果
# print(f'Aiming: {aiming}')
# print(f'Coverage: {coverage}')
# print(f'Accuracy: {accuracy}')
# print(f'Absolute True: {absolute_true}')
# print(f'Absolute False: {absolute_false}')
# # 打印评价指标
# print(f'Decision Tree Accuracy: {accuracy}')
# print('Classification Report:\n', class_report)
# print('Confusion Matrix:\n', conf_matrix)

# # 可视化混淆矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['AMP', 'ACP', 'ADP', 'AHP', 'AIP'], yticklabels=['AMP', 'ACP', 'ADP', 'AHP', 'AIP'])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix Heatmap')
# plt.savefig('confusion_matrix_heatmap.png')
# plt.show()

# # 可视化决策树
# dot_data = export_graphviz(
#     tree,
#     out_file=None,
#     feature_names=['high_level_feature1', 'high_level_feature2', 'high_level_feature3', 'high_level_feature4', 'high_level_feature5'],
#     class_names=['ACP', 'ADP', 'AHP', 'AIP', 'AMP'],
#     filled=True,
#     rounded=True,
#     special_characters=True
# )

# graph = graphviz.Source(dot_data, format='png')
# graph.render('decision_tree_visualization')

# # 打印输出以确认保存路径
# print("Visualization saved to 'decision_tree_visualization.png'")
# print("Confusion Matrix Heatmap saved to 'confusion_matrix_heatmap.png'")

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import scores, evaluate
from sklearn.preprocessing import LabelBinarizer

# 假设特征和标签已从文件加载
with open('model_data.pkl', 'rb') as f:
    model_data = pickle.load(f)

features = model_data['features']
labels = model_data['labels']

# 训练决策树
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(features, labels)
# 使用决策树模型进行预测
tree_predictions = tree.predict(features)

# 如果labels为多分类标签，需要转换为二进制形式
lb = LabelBinarizer()
labels_bin = lb.fit_transform(labels)
tree_predictions_bin = lb.transform(tree_predictions)  # 转换预测结果为二进制标签形式

# 计算性能指标
accuracy = accuracy_score(labels, tree_predictions)
class_report = classification_report(labels, tree_predictions)
conf_matrix = confusion_matrix(labels, tree_predictions)

# 使用您定义的评价函数
aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(tree_predictions_bin, labels_bin)

# 打印您定义的评价指标结果
print(f'Aiming: {aiming}')
print(f'Coverage: {coverage}')
print(f'Accuracy: {accuracy}')
print(f'Absolute True: {absolute_true}')
print(f'Absolute False: {absolute_false}')
# 打印评价指标
print(f'Decision Tree Accuracy: {accuracy}')
print('Classification Report:\n', class_report)
print('Confusion Matrix:\n', conf_matrix)


# 可视化混淆矩阵
plt.figure(figsize=(15, 15))  # 增加图像尺寸
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['AMP', 'ACP', 'ADP', 'AHP', 'AIP'], 
            yticklabels=['AMP', 'ACP', 'ADP', 'AHP', 'AIP'], 
            annot_kws={"size": 30})  # 增大框内数字字体大小
plt.xlabel('Predicted', fontsize=30)  # 增大坐标轴上的标签字体大小
plt.ylabel('True', fontsize=30)  # 增大坐标轴上的标签字体大小
plt.title('Confusion Matrix Heatmap', fontsize=30)  # 增大标题字体大小
plt.xticks(fontsize=25)  # 增大坐标轴上的数字字体大小
plt.yticks(fontsize=25)  # 增大坐标轴上的数字字体大小

# 保存图像为 PNG 和 EPS 格式
plt.savefig('confusion_matrix_heatmap.png')
plt.savefig('confusion_matrix_heatmap.eps', format='eps')

plt.show()
plt.close()

# 输出路径信息，确认保存位置
print("Visualization saved to confusion_matrix_heatmap.png and confusion_matrix_heatmap.eps")


# 可视化决策树
dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=['high_level_feature1', 'high_level_feature2', 'high_level_feature3', 'high_level_feature4', 'high_level_feature5'],
    class_names=['ACP', 'ADP', 'AHP', 'AIP', 'AMP'],
    filled=True,
    rounded=True,
    special_characters=True
)

# 保存为 EPS 格式
graph = graphviz.Source(dot_data, format='eps')
graph.render('decision_tree_visualization', format='eps')

# 保存为 PNG 格式
graph = graphviz.Source(dot_data, format='png')
graph.render('decision_tree_visualization', format='png')

# 打印输出以确认保存路径
print("Visualization saved to 'decision_tree_visualization.eps' and 'decision_tree_visualization.png'")
print("Confusion Matrix Heatmap saved to 'confusion_matrix_heatmap.eps' and 'confusion_matrix_heatmap.png'")
