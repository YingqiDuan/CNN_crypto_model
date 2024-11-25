# 导入必要的库
from sklearn.metrics import classification_report
import methods as m


def main():
    # 设置参数
    file_path = (
        r"merged_csv\BTCUSDT_4h_2022-03-28_to_2024-11-05.csv"  # 替换为你的CSV文件路径
    )
    future_days = 5  # 预测未来5天

    # 获取数据
    df = m.load_data(file_path)

    if df.empty:
        print("DataFrame为空，退出程序。")
        return

    # 特征工程
    df = m.feature_engineering(df)

    # 标签定义
    df = m.define_target(df, future_days=future_days)

    if df.empty:
        print("经过标签定义后，DataFrame为空，退出程序。")
        return

    # 数据分割
    train, val, test = m.split_data(df, train_size=0.7, val_size=0.15)

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("某个数据集为空，退出程序。")
        return

    # 准备特征和标签
    features = ["EMA_short", "EMA_long"]
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

    # 初步训练XGBoost模型
    print("\n开始训练初步的XGBoost模型...")
    initial_model = m.train_xgboost(X_train, y_train, X_val, y_val)

    if initial_model is None:
        print("初步模型训练失败，退出程序。")
        return

    # 超参数调优
    print("\n开始进行超参数调优...")
    best_model = m.hyperparameter_tuning(X_train, y_train)

    if best_model is None:
        print("超参数调优失败，退出程序。")
        return

    # 在验证集上评估最佳模型
    print("\n评估最佳模型在验证集上的性能...")
    m.evaluate_model(best_model, X_val, y_val, dataset_name="Validation")

    # 在测试集上评估最佳模型
    print("\n评估最佳模型在测试集上的性能...")
    m.evaluate_model(best_model, X_test, y_test, dataset_name="Test")

    # 保存最佳模型
    model_filename = "xgboost_ema_model/output/xgboost_ema_model.joblib"
    m.save_model(best_model, model_filename)

    # 加载模型并进行预测（示例）
    loaded_model = m.load_model(model_filename)

    if loaded_model is not None:
        # 使用加载的模型在测试集上进行预测
        y_test_pred_loaded = loaded_model.predict(X_test)
        print("\n使用加载的模型在测试集上的预测结果：")
        print(
            classification_report(
                y_test,
                y_test_pred_loaded,
                labels=[0, 1],
                target_names=["Down", "Up"],
                zero_division=0,
            )
        )

    # 策略回测
    print("\n开始策略回测...")
    m.backtest_strategy(best_model, test, features)


if __name__ == "__main__":
    main()
