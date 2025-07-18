from elml.dataset import ElectrolyteDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def modify_feature_matrix(feature_matrix: list[list[float]]) -> list[float]:
    # 原始矩阵：[类型嵌入（3） + 特定属性（8） + MACCS指纹（166） + 占比（1）]
    # 修改后的矩阵：溶剂特定属性（8） + 锂盐占比（1）
    salt = 0
    solvent_matrix: list[float] = [0, 0, 0, 0, 0, 0, 0]
    for row in feature_matrix:
        if row[0:3] == [1, 0, 0]:  # 锂盐
            salt += row[-1]
        elif row[0:3] == [0, 1, 0]:  # 溶剂
            for i in range(7):
                solvent_matrix[i] += row[3 + i] * row[-1]
    updated_feature_matrix = solvent_matrix + [salt]
    return updated_feature_matrix


dataset = ElectrolyteDataset(dataset_file="data/calisol23.json")

features = []
for formula in dataset.formulas:
    features.append(
        [formula.performance["conductivity"]]
        + modify_feature_matrix(formula.get_feature_matrix())
    )

features_df = pd.DataFrame(features)
# 设置列名
features_df.columns = [
    "电导率",
    "密度",
    "熔点",
    "沸点",
    "介电常数",
    "粘度",
    "偶极化率",
    "电化学窗口",
    "锂盐占比",
]
# 计算皮尔逊相关性矩阵
correlation_matrix = features_df.corr(method="pearson")

# (可选) 打印出相关性矩阵查看具体数值
print(correlation_matrix)

# 设置图表大小
plt.figure(figsize=(12, 10))
# 显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 显示负号
plt.rcParams["axes.unicode_minus"] = False

# 绘制热图
# cmap='RdBu_r' 是一个红-白-蓝的配色方案，红色代表正相关，蓝色代表负相关，非常直观
# annot=True 会在格子里显示数值，如果特征太多导致看不清，可以设为False
# vmin=-1, vmax=1 固定了颜色映射的范围，使得不同图之间具有可比性
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="RdBu_r",
    fmt=".2f",  # 数值格式化，保留两位小数
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)

# 添加标题
plt.title("Feature Correlation Heatmap", fontsize=16)

# 显示图表
plt.show()
