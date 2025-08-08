"""
ElectrolyteWorkflow使用示例
展示如何使用统一的工作流类进行完整的机器学习流水线
"""

from elml.dataset import ElectrolyteDataset
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
        num_workers=0,
        lr=1e-4,
        num_epochs=500,
        early_stopping_patience=40,
        data_path="data/calisol23/calisol231.json",
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
        "num_workers": 0,
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


def custom_mlp_analysis():
    """自定义分析的示例"""

    config = WorkflowConfig(
        input_dim=179,
        model_type="mlp",
        data_path="data/calisol23/calisol23.json",
        model_name="mlp_model_12_splits_custom_25",
    )

    workflow = ElectrolyteWorkflow(config)

    # 假设已经有训练好的模型
    try:
        # 仅执行预测和分析部分
        workflow.setup_data()
        workflow.setup_model()
        workflow.setup_predictor(str(workflow.log_dir / "best_model.pth"))

        # 使用不同的数据集进行分析
        all_dataset = ElectrolyteDataset(feature_mode="weighted_average")
        new_dataset = ElectrolyteDataset(feature_mode="weighted_average")

        dataset_liasf6 = ElectrolyteDataset(
            "data/calisol23/calisol23_liAsf6.json", feature_mode="weighted_average"
        )
        dataset_litfsi = ElectrolyteDataset(
            "data/calisol23/calisol23_litfsi.json", feature_mode="weighted_average"
        )

        dataset_test = workflow.datasets["test"]

        dataset_tem = ElectrolyteDataset(
            "data/calisol23/calisol23_tem.json", feature_mode="weighted_average"
        )

        dataset_fraction = ElectrolyteDataset(
            "data/calisol23/calisol23_fraction.json", feature_mode="weighted_average"
        )

        # 合并数据集
        # for e in dataset_liasf6:
        #     all_dataset.add_formula(e)
        #     new_dataset.add_formula(e)
        for e in dataset_litfsi:
            all_dataset.add_formula(e)
            new_dataset.add_formula(e)
        for e in dataset_test:
            all_dataset.add_formula(e)

        # # 这里可以传入自定义的数据集
        # analysis_results = workflow.analyze(
        #     dataset=all_dataset,
        #     # show_plots=True,
        #     save_plots=True,
        #     plot_filename="custom_analysis_all.png",
        # )
        # # 保存数据到文件
        # with open("custom_analysis_all.json", "w") as f:
        #     import json

        #     json.dump(analysis_results, f, indent=4)
        # # 这里可以传入自定义的数据集
        # analysis_results = workflow.analyze(
        #     dataset=new_dataset,
        #     # show_plots=True,
        #     save_plots=True,
        #     plot_filename="custom_analysis_new_salt.png",
        # )
        # # 保存数据到文件
        # with open("custom_analysis_new_salt.json", "w") as f:
        #     import json

        #     json.dump(analysis_results, f, indent=4)
        # # 这里可以传入自定义的数据集
        # analysis_results = workflow.analyze(
        #     dataset=dataset_test,
        #     # show_plots=True,
        #     save_plots=True,
        #     plot_filename="custom_analysis.png",
        # )
        # # 保存数据到文件
        # with open("custom_analysis.json", "w") as f:
        #     import json

        #     json.dump(analysis_results, f, indent=4)
        # 这里可以传入自定义的数据集
        analysis_results = workflow.analyze(
            dataset=dataset_tem,
            # show_plots=True,
            save_plots=True,
            plot_filename="custom_analysis_tem.png",
        )
        # 保存数据到文件
        with open("custom_analysis_tem.json", "w") as f:
            import json

            json.dump(analysis_results, f, indent=4)
        # 这里可以传入自定义的数据集
        analysis_results = workflow.analyze(
            dataset=dataset_fraction,
            # show_plots=True,
            save_plots=True,
            plot_filename="custom_analysis_fraction.png",
        )
        # 保存数据到文件
        with open("custom_analysis_fraction.json", "w") as f:
            import json

            json.dump(analysis_results, f, indent=4)
        print("自定义分析完成")
        if analysis_results:
            print(f"R²得分: {analysis_results['r2']:.4f}")
        else:
            print("分析结果为空")

    except Exception as e:
        print(f"自定义分析失败: {e}")


def custom_transformer_analysis():
    """自定义分析的示例"""

    config = WorkflowConfig(
        input_dim=179,
        model_type="transformer",
        data_path="data/calisol23/calisol23.json",
        model_name="transformer_model_12_splits_custom_25",
    )

    workflow = ElectrolyteWorkflow(config)

    # 假设已经有训练好的模型
    try:
        # 仅执行预测和分析部分
        workflow.setup_data()
        workflow.setup_model()
        workflow.setup_predictor(str(workflow.log_dir / "best_model.pth"))

        # 使用不同的数据集进行分析
        all_dataset = ElectrolyteDataset(feature_mode="sequence")
        new_dataset = ElectrolyteDataset(feature_mode="sequence")
        dataset_liasf6 = ElectrolyteDataset(
            "data/calisol23/calisol23_liAsf6.json", feature_mode="sequence"
        )
        dataset_litfsi = ElectrolyteDataset(
            "data/calisol23/calisol23_litfsi.json", feature_mode="sequence"
        )

        dataset_test = workflow.datasets["test"]

        dataset_tem = ElectrolyteDataset(
            "data/calisol23/calisol23_tem.json", feature_mode="sequence"
        )

        dataset_fraction = ElectrolyteDataset(
            "data/calisol23/calisol23_fraction.json", feature_mode="sequence"
        )

        # 合并数据集
        # for e in dataset_liasf6:
        #     all_dataset.add_formula(e)
        #     new_dataset.add_formula(e)
        for e in dataset_litfsi:
            all_dataset.add_formula(e)
            new_dataset.add_formula(e)
        for e in dataset_test:
            all_dataset.add_formula(e)

        # 这里可以传入自定义的数据集
        analysis_results = workflow.analyze(
            dataset=dataset_tem,
            # show_plots=True,
            save_plots=True,
            plot_filename="custom_analysis_tem.png",
        )
        # 保存数据到文件
        with open("custom_analysis_tem.json", "w") as f:
            import json

            json.dump(analysis_results, f, indent=4)
        # 这里可以传入自定义的数据集
        analysis_results = workflow.analyze(
            dataset=dataset_fraction,
            # show_plots=True,
            save_plots=True,
            plot_filename="custom_analysis_fraction.png",
        )
        # 保存数据到文件
        with open("custom_analysis_fraction.json", "w") as f:
            import json

            json.dump(analysis_results, f, indent=4)

        # # 这里可以传入自定义的数据集
        # analysis_results = workflow.analyze(
        #     dataset=all_dataset,
        #     # show_plots=True,
        #     save_plots=True,
        #     plot_filename="custom_analysis_all.png",
        # )

        # # 保存数据到文件
        # with open("custom_analysis_all.json", "w") as f:
        #     import json

        #     json.dump(analysis_results, f, indent=4)

        # # 这里可以传入自定义的数据集
        # analysis_results = workflow.analyze(
        #     dataset=new_dataset,
        #     # show_plots=True,
        #     save_plots=True,
        #     plot_filename="custom_analysis_new_salt.png",
        # )
        # # 保存数据到文件
        # with open("custom_analysis_new_salt.json", "w") as f:
        #     import json

        #     json.dump(analysis_results, f, indent=4)
        # # 这里可以传入自定义的数据集
        # analysis_results = workflow.analyze(
        #     dataset=dataset_test,
        #     # show_plots=True,
        #     save_plots=True,
        #     plot_filename="custom_analysis.png",
        # )
        # # 保存数据到文件
        # with open("custom_analysis.json", "w") as f:
        #     import json

        #     json.dump(analysis_results, f, indent=4)
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
    # custom_mlp_analysis()
    custom_transformer_analysis()
