import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from Config.global_config import *
from Config.ml_strategy_config import *


def filters(df, filter_name):

    if filter_name == "小市值":
        df = df[df['总市值 （万元）'] < 300000]

    else:
        raise Exception("尚无此过滤方法。")

    return df


def build_prediction_data_set(pick_stock_df_interval, feature_li, batch_size, data_filter):
    df = pick_stock_df_interval

    # 删除缺失值
    df.dropna(axis=0, how="any", inplace=True)

    # 过滤数据
    df = filters(df, data_filter)

    # 选择特征
    X = df[feature_li].values

    # 将数据转换为 PyTorch 张量
    X_tensor = torch.tensor(X).float()

    # 创建 PyTorch 数据集
    prediction_dataset = TensorDataset(X_tensor)

    # 创建数据加载器
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)

    return prediction_loader


def load_model(model_path):
    """
    Load the saved model from the given path.
    """
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, data_loader):
    """
    Use the model to make predictions on the data provided by data_loader.
    """
    predictions = []
    with torch.no_grad():  # No need to track gradients
        for inputs, _ in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.tolist())  # Convert to list and store
    return predictions


def ML_model_predictor(pick_stock_df, period_type, feature_li, data_filter):

    for prd_interval in model_time_pair_dct.keys():
        pick_stock_df_interval = pick_stock_df[(pick_stock_df['交易日期'] >= pd.to_datetime(prd_interval[0]))
                                      & (pick_stock_df['交易日期'] <= pd.to_datetime(prd_interval[1]))]
        if len(pick_stock_df_interval) == 0:
            continue

        # Prepare your data loader for the data you want to predict
        prediction_loader = build_prediction_data_set(pick_stock_df_interval, feature_li, 64, data_filter)

        # 用对应的训练时间构建路径
        train_interval = model_time_pair_dct[prd_interval]
        model_path = r'{}\strategy\pick_stock\model_utils\saved_model\FCN_reg_{}_{}-{}.pth'.format(project_path, train_interval[0], train_interval[1], period_type)
        model = load_model(model_path)

        # Make predictions
        predictions = predict(model, prediction_loader)

        pick_stock_df.loc[(pick_stock_df['交易日期'] >= pd.to_datetime(prd_interval[0]))
                          & (pick_stock_df['交易日期'] <= pd.to_datetime(prd_interval[1])), "机器学习预测值"] = predictions

    return pick_stock_df
