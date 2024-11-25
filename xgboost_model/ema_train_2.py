import methods as m
import numpy as np
import xgboost as xgb


def main():
    # 设置参数
    original_file_path = ""  # 原始CSV文件路径
    new_file_path = "D:\crypto_ml_project\My Project\merged_csv\BTCDOMUSDT_4h_2023-11-23_to_2024-11-20.csv"  # 新收集的CSV文件路径
    model_filename = "xgboost_ema_model.joblib"  # 模型文件名
    future_days = 5  # 预测未来5天

    # 1. 加载原始数据和新数据
    original_df = m.load_data(original_file_path)
    new_df = m.load_data(new_file_path)

    if original_df.empty and new_df.empty:
        print("原始数据和新数据均为空，退出程序。")
        return
    elif original_df.empty:
        combined_df = new_df
    elif new_df.empty:
        combined_df = original_df
    else:
        combined_df = m.merge_data(original_df, new_df)

    # 2. 特征工程
    combined_df = m.feature_engineering(combined_df)

    # 3. 标签定义
    combined_df = m.define_target(combined_df, future_days=future_days)

    if combined_df.empty:
        print("经过标签定义后，DataFrame为空，退出程序。")
        return

    # 4. 数据分割
    train, val, test = m.split_data(combined_df, train_size=0.7, val_size=0.15)

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("某个数据集为空，退出程序。")
        return

    # 5. 准备特征和标签
    features = [
        "EMA_short",
        "EMA_long",
    ]
    for feature in features:
        if feature not in train.columns:
            print(f"特征'{feature}'不存在于DataFrame中。")
            return

    X_train = train[features]
    y_train = train["Target"]
    X_val = val[features]
    y_val = val["Target"]
    X_test = test[features]
    y_test = test["Target"]

    # 检查特征和标签是否为空
    if (
        X_train.empty
        or y_train.empty
        or X_val.empty
        or y_val.empty
        or X_test.empty
        or y_test.empty
    ):
        print("特征或标签为空，无法继续。")
        return

    # 6. 加载已保存的模型（如果存在）
    loaded_model = m.load_model(model_filename)
    if loaded_model is not None:
        print("加载已保存的模型，准备继续训练。")
        # 计算scale_pos_weight
        from sklearn.utils import class_weight

        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        scale_pos_weight = class_weights[0] / class_weights[1]

        # 继续训练模型，使用之前的模型作为基准
        loaded_model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=loaded_model.get_params()["n_estimators"]
            + 100,  # 增加更多的树
            learning_rate=loaded_model.get_params()["learning_rate"],
            max_depth=loaded_model.get_params()["max_depth"],
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )

        # 训练模型
        loaded_model.fit(X_train, y_train)

        # 评估模型
        print("\n评估继续训练后的模型在验证集上的性能...")
        m.evaluate_model(loaded_model, X_val, y_val, dataset_name="Validation")

        # 保存更新后的模型
        m.save_model(loaded_model, model_filename)
    else:
        print("未找到已保存的模型，开始训练新模型。")
        # 计算scale_pos_weight
        from sklearn.utils import class_weight

        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        scale_pos_weight = class_weights[0] / class_weights[1]

        # 训练初步模型
        model = m.train_xgboost(
            X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos_weight
        )

        if model is None:
            print("初步模型训练失败，退出程序。")
            return

        # 保存模型
        m.save_model(model, model_filename)

    # 7. 重新加载更新后的模型
    updated_model = m.load_model(model_filename)

    if updated_model is not None:
        # 在测试集上评估模型
        print("\n评估更新后的模型在测试集上的性能...")
        m.evaluate_model(updated_model, X_test, y_test, dataset_name="Test")

        # 策略回测
        print("\n开始策略回测...")
        m.backtest_strategy(updated_model, test, features)
    else:
        print("加载更新后的模型失败，无法进行测试和回测。")


if __name__ == "__main__":
    main()
