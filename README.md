# ElectrolyteML: 基于机器学习的电池电解液性能预测框架

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

**ElectrolyteML** 是一个专为电池材料科学设计的开源研究工具包，它利用先进的机器学习技术，特别是Transformer模型，来预测锂离子电池电解液的宏观性能（如离子电导率、粘度等），旨在加速新型高性能电解液配方的设计与筛选过程。

本框架的核心思想是将电解液的化学组分（盐、溶剂、添加剂）及其配比，通过分子指纹和物化性质进行深度特征表示，并构建一个端到端的工作流，实现从数据处理、模型训练到性能预测的全自动化，为材料研发提供强大的数据驱动支持。

## 核心特性

-   **🧪 结构化的材料库**: 内置常用电解液材料（锂盐、溶剂、添加剂）的物化属性数据库，支持轻松查询与扩展。
-   **🧬 智能化的特征工程**: 自动将化学配方转换为机器学习模型可理解的特征向量，融合了分子结构指纹与宏观物理化学性质。
-   **🤖 先进的预测模型**: 采用基于PyTorch实现的Transformer架构，能有效捕捉配方中不同组分间的复杂相互作用。
-   **⚙️ 端到端自动化工作流**: 提供从数据加载、模型训练、验证到最终预测的完整、高度封装的流程，用户只需几行代码即可完成整个实验。
-   **🧩 模块化与高扩展性设计**: 项目代码遵循高内聚、低耦合原则，分为材料、电池、数据集和模型四大核心模块，易于维护与功能扩展。

## 📦 安装指南

### 环境依赖

- Python 3.12+
- PyTorch
- RDKit
- Pydantic
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (推荐, 高速Python包管理器) 或 `pip`

### 安装步骤

**使用 `uv` (推荐):**

```bash
# 克隆本项目
git clone https://github.com/your-username/ElectrolyteML.git
cd ElectrolyteML

# 使用uv同步环境，它会自动创建虚拟环境并安装所有依赖
uv sync
```

**使用 `pip`:**

```bash
# 克隆本项目
git clone https://github.com/your-username/ElectrolyteML.git
cd ElectrolyteML

# 手动创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# 安装依赖
pip install -e .
```

## 工作流快速上手

下面的示例将引导您完成一个完整的端到端任务：**加载数据集 -> 训练模型 -> 定义新配方 -> 预测性能**。

```python
"""
ElectrolyteWorkflow使用示例
展示如何使用统一的工作流类进行完整的机器学习流水线
"""

from elml.ml import ElectrolyteWorkflow, WorkflowConfig


def example_transformer_workflow():
    """Transformer模型的完整工作流示例"""

    # 方式1: 使用配置对象
    config = WorkflowConfig(
        model_type="transformer",
        input_dim=179,
        model_dim=256,
        nhead=8,
        num_encoder_layers=4,
        feature_mode="sequence",  # transformer使用序列特征
        batch_size=256,
        num_workers=16,
        lr=1e-4,
        num_epochs=500,
        early_stopping_patience=40,
        data_path="data/calisol23/calisol23.json",
        log_dir="runs",
        model_name="transformer_model_4",
    )

    # 创建工作流
    workflow = ElectrolyteWorkflow(config)

    # 运行完整流水线
    results = workflow.run_full_pipeline()

    print(f"工作流状态: {results['status']}")
    if "analysis" in results:
        print(f"模型R²得分: {results['analysis']['r2']:.4f}")
        print(f"平均绝对误差: {results['analysis']['mae']:.4f}")


def example_mlp_workflow():
    """MLP模型的完整工作流示例"""

    # 方式2: 使用配置字典
    config_dict = {
        "model_type": "mlp",
        "input_dim": 179,
        "dropout_rate": 0.3,
        "feature_mode": "weighted_average",  # mlp使用加权平均特征
        "batch_size": 256,
        "num_workers": 16,
        "lr": 1e-4,
        "optimizer_name": "adam",
        "num_epochs": 500,
        "early_stopping_patience": 40,
        "data_path": "data/calisol23/calisol23.json",
        "log_dir": "runs",
        "model_name": "mlp_model_4",
    }

    # 创建工作流
    workflow = ElectrolyteWorkflow(config_dict)

    # 运行完整流水线
    results = workflow.run_full_pipeline()

    print(f"工作流状态: {results['status']}")
    if "analysis" in results:
        print(f"模型R²得分: {results['analysis']['r2']:.4f}")
        print(f"平均绝对误差: {results['analysis']['mae']:.4f}")


def example_step_by_step_workflow():
    """分步骤执行的工作流示例"""

    config = WorkflowConfig(
        model_type="transformer",
        data_path="data/calisol23/calisol23.json",
        model_name="step_by_step_model",
        num_epochs=3,
        feature_mode="sequence",  # transformer使用序列特征
    )

    workflow = ElectrolyteWorkflow(config)

    # 1. 设置数据
    train_ds, val_ds, test_ds = workflow.setup_data()

    # 2. 设置模型
    workflow.setup_model()

    # 3. 查看模型信息
    model_info = workflow.get_model_info()
    print(f"模型参数数量: {model_info['total_parameters']:,}")

    # 4. 设置训练器并训练
    workflow.setup_trainer()
    training_results = workflow.train()

    # 5. 设置预测器
    workflow.setup_predictor()

    # 6. 进行预测
    predictions, actual_values = workflow.predict()

    # 7. 分析结果
    analysis_results = workflow.analyze(save_plots=True, show_plots=False)

    print(f"训练完成，最佳验证损失: {training_results.get('best_val_loss', 'N/A')}")
    if analysis_results:
        print(f"模型R²得分: {analysis_results['r2']:.4f}")
    else:
        print("分析结果为空")


def example_load_and_predict():
    """加载已训练模型进行预测的示例"""

    # 使用已有的配置和模型
    workflow = ElectrolyteWorkflow(WorkflowConfig())

    # 从检查点加载
    try:
        workflow.load_from_checkpoint("runs/transformer_model")

        # 设置数据
        workflow.setup_data("data/calisol23/calisol23.json")

        # 设置预测器
        workflow.setup_predictor()

        # 进行分析
        results = workflow.analyze()
        if results:
            print(f"加载的模型R²得分: {results['r2']:.4f}")
        else:
            print("分析结果为空")

    except FileNotFoundError:
        print("未找到已训练的模型，请先运行训练示例")


def example_config_from_file():
    """从配置文件创建工作流的示例"""

    # 首先创建一个配置文件
    config = WorkflowConfig(
        model_type="mlp",
        data_path="data/calisol23/calisol23.json",
        model_name="config_file_model",
        num_epochs=3,
        feature_mode="weighted_average",  # mlp使用加权平均特征
    )

    # 创建临时工作流以保存配置
    temp_workflow = ElectrolyteWorkflow(config)
    config_path = temp_workflow.log_dir / "config.json"

    print(f"配置文件已保存到: {config_path}")

    # 从配置文件创建新的工作流
    new_workflow = ElectrolyteWorkflow(str(config_path))

    # 运行流水线（跳过训练，仅演示加载）
    try:
        new_workflow.run_full_pipeline(skip_training=True)
    except Exception as e:
        print(f"跳过训练时需要指定已有模型: {e}")


def example_custom_analysis():
    """自定义分析的示例"""

    config = WorkflowConfig(
        model_type="mlp",
        data_path="data/calisol23/calisol23.json",
        model_name="mlp_model_4",
    )

    workflow = ElectrolyteWorkflow(config)

    # 假设已经有训练好的模型
    try:
        # 仅执行预测和分析部分
        workflow.setup_data()
        workflow.setup_model()
        workflow.setup_predictor(str(workflow.log_dir / "best_model.pth"))

        # 使用不同的数据集进行分析
        # 这里可以传入自定义的数据集
        analysis_results = workflow.analyze(
            show_plots=True, save_plots=True, plot_filename="custom_analysis.png"
        )

        print("自定义分析完成")
        if analysis_results:
            print(f"R²得分: {analysis_results['r2']:.4f}")
        else:
            print("分析结果为空")

    except Exception as e:
        print(f"自定义分析失败: {e}")


if __name__ == "__main__":
    print("ElectrolyteWorkflow 使用示例")
    print("=" * 50)

    # 选择要运行的示例
    print("1. Transformer模型完整工作流")
    # example_transformer_workflow()

    print("\n2. MLP模型完整工作流")
    # example_mlp_workflow()

    print("\n3. 分步骤执行工作流")
    # example_step_by_step_workflow()

    print("\n4. 加载已训练模型")
    # example_load_and_predict()

    print("\n5. 从配置文件创建工作流")
    # example_config_from_file()

    print("\n6. 自定义分析")
    example_custom_analysis()

```

## 模块化API详解

项目代码高度模块化，核心功能由以下四个模块提供：

### 1. `elml.material` - 材料库模块

-   **功能说明**: 此模块是整个框架的基石。它负责从 `materials.json` 文件中加载所有化学物质（盐、溶剂、添加剂）的物理化学性质，并构建一个全局唯一的材料数据库实例 `MLibrary`，以便在项目各处进行高效、统一的调用。这种设计避免了重复创建材料对象，节约了内存。
-   **核心API**: `elml.MLibrary` (全局单例)
    -   `MLibrary.get_material(abbr: str)`: 通过材料的缩写（如 "LiPF6"）获取材料对象。
    -   `MLibrary.has_material(abbr: str)`: 检查材料是否存在于库中。
-   **使用示例**:
    ```python
    from elml import MLibrary
    
    # MLibrary 在程序启动时已自动加载完毕
    print(f"材料库中共有 {len(MLibrary)} 种材料。")
    
    # 获取溶剂 Ethylene Carbonate (EC)
    ec_solvent = MLibrary.get_material("EC")
    
    # 访问其物理化学性质
    print(f"材料名称: {ec_solvent.name}")
    print(f"CAS号: {ec_solvent.cas_registry_number}")
    print(f"分子量: {ec_solvent.molecular_weight} g/mol")
    print(f"介电常数: {ec_solvent.dielectric_constant}")
    ```

### 2. `elml.battery` - 电池与电解液定义模块

-   **功能说明**: 此模块定义了电解液和电池的面向对象模型。核心是 `Electrolyte` 类，它允许用户通过组合来自 `material` 模块的各种材料来精确构建一个电解液配方。
-   **核心API**: `elml.battery.Electrolyte`
    -   `Electrolyte.create(...)`: 一个工厂方法，用于方便地创建 `Electrolyte` 实例。
    -   `electrolyte.show()`: 打印出格式化的配方详情。
    -   `electrolyte.to_feature_vector()`: 将配方转换为模型可用的特征向量（内部调用）。
-   **使用示例**:
    ```python
    from elml.battery import Electrolyte
    from elml import MLibrary
    
    # 使用 create 方法定义一个复杂的电解液配方
    complex_electrolyte = Electrolyte.create(
        id="complex-electrolyte-1",
        description="A complex electrolyte formulation with multiple components.",
        name="Complex-Recipe-1",
        salts=[(MLibrary.get_material("LiPF6"), 15.0)],
        solvents=[
            (MLibrary.get_material("EC"), 40.0),
            (MLibrary.get_material("EMC"), 60.0),
        ],
        additives=[(MLibrary.get_material("VC"), 2.0)],
        performance={},
    )
    
    # 显示配方信息
    complex_electrolyte.show()
    ```

### 3. `elml.dataset` - 数据集与工作流模块

- **功能说明**: 这是执行机器学习任务的**主控制模块**。它将数据加载、预处理等一系列复杂操作封装在 `ElectrolyteDataset` 类中，提供了极其简洁的高级API。

-   **核心API**: `elml.dataset.ElectrolyteDataset`
    -   `ElectrolyteDataset(dataset_file)`: 初始化，加载JSON格式的数据集。
    
-   **使用示例**: (此示例聚焦于训练和加载)
    
    ```python
    from elml.dataset import ElectrolyteDataset
    
    # 初始化数据集对象
    dataset = ElectrolyteDataset("data/calisol23/calisol23.json")
    # 打印数据集的基本信息
    print("数据集大小:", len(dataset))
    ```

### 4. `elml.ml` - 机器学习模型模块

- **功能说明**: 此模块包含了机器学习模型的核心实现。通常，用户不需要直接与此模块交互，而是通过 `workflow` 模块的API来间接使用它们。

-   **核心API**: 
    
    -   `elml.ml.models.ElectrolyteTransformer`: Transformer模型类。
    -   `elml.ml.predictor.ElectrolyteMLP`: MLP模型类。
    
    

##  📁 项目结构

```
ElectrolyteML/
├── elml/              # 核心源码包
│   ├── material/      # 材料库管理模块
│   ├── battery/       # 电解液与电池定义模块
│   ├── dataset/       # 数据集与工作流控制模块
│   └── ml/            # 机器学习模型实现模块
├── data/              # 存放数据集文件 (如 .json, .csv)
├── examples/          # 存放使用示例脚本
├── runs/              # 训练过程中产生的模型、日志等文件的默认输出目录
├── pyproject.toml     # 项目配置文件及依赖
└── README.md          # 本文档
```

## 🤝 贡献指南

我们非常欢迎任何形式的贡献，无论是新功能开发、Bug修复、文档改进还是问题报告。请随时提交Pull Request或开启Issue。

## 📜 许可证

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。
