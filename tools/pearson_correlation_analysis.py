from elml.dataset import ElectrolyteDataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = ElectrolyteDataset(
    dataset_file="data/calisol23/calisol23.json", feature_mode="weighted_average"
)

columns = [
    "电导率",
    "密度",
    "熔点",
    "沸点",
    "介电常数",
    "粘度",
    "偶极化率",
    "电化学窗口",
    "氢键能",
    "温度",
    "锂盐占比",
]
features_df = pd.DataFrame(columns=columns)
# 设置列名

for i in range(len(dataset)):
    features_tensor, temperature_tensor, target_tensor = dataset[i]
    features = features_tensor.numpy().tolist()
    temperature = temperature_tensor.numpy().tolist()
    target = target_tensor.numpy().tolist()

    # 获取锂盐浓度
    salt_concentration = dataset.formulas[i].proportions[0] ** 0.01

    features_df.loc[i] = [target] + features[-9:] + [salt_concentration]

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
