# 基于机器学习的电池电解液性能预测与配方优化系统

## 1. 项目概述

本项目是一个结合材料科学与人工智能技术的研究工具，专注于锂离子电池电解液性能预测与配方优化。该项目利用分子特征提取、材料属性标准化和深度学习模型，基于材料的基础物性，实现电解液性能的精准预测，为电池研发提供高效的计算辅助工具。

### 功能特点

- **电解液材料库**
  - 支持锂盐、溶剂、添加剂三大类材料管理
  - 基于RDKit的分子结构特征自动提取
  - 材料属性标准化和范围限制
  - 内置数据库包含常用电解液材料物性数据
- **电解液配方系统**
  - 灵活的组分配比和材料组合
  - 基于占比的配方特征向量生成
  - 配方序列化和反序列化支持
- **电解液性能预测**
  - 基于Transformer架构的电解液电导率、粘度预测
  - 融合分子指纹、物理化学性质的多模态特征学习
  - 训练-验证-测试完整评估流程
- **电池整体性能预测**
  - 可扩展分析电池正负极材料
  - 综合电解液配方、正负极材料对电池整体性能进行预测
  - 实现通过材料物性直接预测电池整体性能
- **可视化与分析**
  - 配方性能比较与分析
  - 结构-性能关系可视化
  - 实验数据与预测结果对比

### 技术实现

- **材料表示**: 使用MACCS分子指纹和物理化学性质
- **内存优化**: 采用对象池模式避免重复材料实例
- **模型架构**: 基于Transformer的序列特征学习
- **数据验证**: 使用Pydantic进行严格的数据类型检查
- **模块解耦**：对电池-电解液-材料三部分进行解耦

## 2. 安装指南

### 环境要求

- Python 3.12+
- PyTorch 1.8+
- RDKit
- Pydantic 2.0+

### 安装步骤

**使用 `uv` (推荐):**

```bash
# uv 是一个极速的 Python 包安装和解析器
# https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

**使用 `pip`:**
```bash
pip install -e .
```

## 3. 项目结构与模块功能

项目采用模块化设计，各主要文件夹功能如下：

```
ElectrolyteML/
├── battery/         # 核心: 定义电池、电解液等数据结构和行为
├── material/        # 核心: 管理材料库（盐、溶剂、添加剂）
├── dataset/         # 核心: 负责数据集加载、模型训练和性能预测
├── ml/              # 核心: 定义机器学习模型（如 Transformer）
├── data/            # 辅助: 存放原始数据文件（.csv, .json）
├── config/          # 辅助: 项目配置文件
├── tools/           # 辅助: 数据处理等工具脚本
└── utils/           # 辅助: 通用辅助函数
```

---

## 4. 核心模块使用说明

以下是各核心模块的详细功能和使用示例。

### `material`: 材料库管理

**功能**:
此模块负责加载和管理所有化学材料（锂盐、溶剂、添加剂）。它从 `materials.json` 文件中读取材料的基础物理化学性质，并提供一个全局实例 `MLibrary` 方便在项目中随时调用。

**关键组件**:
- `Material`: 单个材料的基类。
- `MaterialLibrary`: 材料数据库，负责加载、查询和管理所有材料。
- `MLibrary`: `MaterialLibrary` 的一个全局单例，预加载了所有材料数据，供项目全局调用。

**使用示例**:

```python
from elml import MLibrary

# MLibrary 已自动加载 'data/materials.json' 中的数据
print(f"材料库加载完成，共 {len(MLibrary)} 种材料。")

# 1. 通过材料缩写 (abbr) 获取材料
lipf6 = MLibrary.get_material("LiPF6", cas_registry_number="21324-40-3")
print(f"获取材料: {lipf6.name}, CAS号: {lipf6.cas_registry_number}")

# 2. 通过 CAS 注册号获取材料
ec = MLibrary.get_material("ec", cas_registry_number="96-49-1")
print(f"获取材料: {ec.name}, 分子量: {ec.molecular_weight}")

# 3. 查看材料属性
print(f"EC 的介电常数: {ec.dielectric_constant}")
print(f"LiPF6 的密度: {lipf6.density} g/cm³")
```

### `battery`: 电解液与电池定义

**功能**:
此模块用于定义电解液配方和完整的电池结构。`Electrolyte` 类是核心，它使用 `material` 模块中的材料来构建具有特定组分和比例的电解液。

**关键组件**:
- `Component`: 表示配方中的一个组分（如特定比例的溶剂）。
- `Electrolyte`: 核心类，由多个 `Component` 组成，代表一个完整的电解液配方。
- `Battery`: （可扩展）用于集成电解液、正极和负极，以进行更全面的电池性能分析。

**使用示例**:

```python
from elml.battery import Electrolyte

# 1. 使用 `Electrolyte.create` 方法轻松创建新配方
#    该方法会自动从 MLibrary 中查找并链接材料
lp30_recipe = Electrolyte.create(
    name="LP30",
    id="standard-lp30",
    description="一个标准的 LiPF6 在 EC/DMC 中的电解液配方",
    salts=[
        # 对于盐，`overall_fraction` 是其在总溶剂质量中的重量百分比
        {"abbr": "LiPF6", "cas_registry_number": "21324-40-3", "overall_fraction": 10.0}
    ],
    solvents=[
        # 对于溶剂，`relative_fraction` 是其在所有溶剂中的相对体积或重量比
        {"abbr": "EC", "cas_registry_number": "96-49-1", "relative_fraction": 50.0},
        {"abbr": "DMC", "cas_registry_number": "616-38-6", "relative_fraction": 50.0},
    ],
    additives=[],  # 也可以添加添加剂
    performance={},
)

# 2. 显示配方详情
lp30_recipe.show()

# 3. 访问配方组分
print(f"配方 '{lp30_recipe.name}' 的盐: {lp30_recipe.salts[0].name}")

```

### `dataset`: 数据集与模型训练

**功能**:
这是执行机器学习任务的入口。它负责加载电解液数据集（包含配方和对应的实验性能），处理数据，训练模型，并使用训练好的模型进行性能预测。

**关键组件**:
- `ElectrolyteDataset`: 核心类，封装了数据加载、预处理、模型训练和预测的所有功能。

**使用示例**:
```python
from elml.dataset import ElectrolyteDataset
from elml.battery import Electrolyte  # 用于创建待预测的配方

# 1. 加载一个电解液数据集（例如，包含多个配方及其电导率）
dataset = ElectrolyteDataset("data/calisol23.json")
print(f"数据集加载成功，包含 {len(dataset)} 个电解液样本。")

# 2. 训练一个性能预测模型
#    模型可以是 'transformer' 或 'mlp'
history = dataset.train_model(
    model="transformer",
    model_config={
        "hidden_dim": 128,
        "n_layers": 2,
        "n_heads": 2,
    },  # Transformer特定参数
    epochs=50,  # 训练轮次
    save_path="models/best_conductivity_model.pth",  # 保存最佳模型
)
print("模型训练完成！")
```

### `ml`: 机器学习模型定义

**功能**:
此模块包含了用于性能预测的神经网络模型的具体实现（基于 PyTorch）。这些模型通常不是直接使用的，而是由 `ElectrolyteDataset` 在 `train_model` 方法内部调用。

**关键组件**:
- `ElectrolyteTransformer`: 基于 Transformer 架构的模型，擅长捕捉序列特征。
- `ElectrolyteMLP`: 一个标准的多层感知器（MLP）模型。

**使用说明**:
您可以在调用 `dataset.train_model` 时，通过 `model` 和 `model_config` 参数来选择和配置这些模型。如果您想实现新的模型架构，可以在此文件夹下添加新的模型类。

---

## 5. 端到端工作流示例

这个例子将展示如何从零开始，结合所有模块，完成一个典型的任务：**创建一个新的电解液配方，并使用训练好的模型预测其离子电导率**。

```python
import torch
from elml.material import MLibrary
from elml.battery import Electrolyte
from elml.dataset import ElectrolyteDataset

# --- 步骤 1: 初始化环境 ---
# 加载材料库 (由 material 模块自动完成)
print(f"✅ 1. 材料库已加载，共 {len(MLibrary)} 种材料。")

# --- 步骤 2: 创建一个新的电解液配方 ---
# 假设我们要设计一个高浓度电解液
my_new_recipe = Electrolyte.create(
    name="My-Custom-High-Concentration-Electrolyte",
    salts=[{"abbr": "LiPF6", "overall_fraction": 15.0}], # 盐浓度提高到 15%
    solvents=[
        {"abbr": "EC", "relative_fraction": 40.0},
        {"abbr": "DMC", "relative_fraction": 60.0}
    ],
    additives=[{"abbr": "VC", "overall_fraction": 2.0}] # 添加 2% 的 VC 添加剂
)
print(f"✅ 2. 已创建新配方: '{my_new_recipe.name}'")
my_new_recipe.show()


# --- 步骤 3: 加载数据集并训练模型 ---
# 使用 calisol23 数据集进行训练
dataset = ElectrolyteDataset("data/calisol23.json")

print("
✅ 3. 开始训练性能预测模型...")
dataset.train_model(
    model="transformer",
    model_config={"hidden_dim": 256, "n_layers": 2, "n_heads": 4},
    epochs=100, # 实际应用中可能需要更多轮次
    learning_rate=0.001,
    save_path="models/best_model.pth"
)
print("模型训练完毕！")


# --- 步骤 4: 预测新配方的性能 ---
# 使用刚刚训练好的模型进行预测
print(f"
✅ 4. 预测新配方 '{my_new_recipe.name}' 的性能...")
predicted_performance = dataset.predict(my_new_recipe)

print("
" + "="*50)
print(f"🎉 预测结果 🎉")
print(f"配方: {my_new_recipe.name}")
print(f"预测的离子电导率: {predicted_performance:.4f} mS/cm")
print("="*50)

```
