import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from Config.global_config import *


def build_prediction_data_set(feature_li, start_date, end_date, period_type, batch_size):
    # 加载数据
    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)
    df = df[feature_li]  # 只选取我们需要的特征，避免排除数据时过度排除

    # 根据日期过滤数据
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]

    # 删除缺失值
    df.dropna(axis=0, how="any", inplace=True)

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


def ML_model_predictor(pick_stock_df, period_type, feature_li):
    # 根据实际情况选择训练时间，每个模型只能用来预测一年，因此模型的数量必须等于回测的年份，也可以考虑增加模型更新频率，说不定会更好
    model_time_pair_dct = {('2010-01-01', '2010-12-31'): ('2007-01-01', '2009-12-31')}

    for prd_time_pair in model_time_pair_dct.keys():
        pick_stock_df_interval = pick_stock_df[(pick_stock_df['交易日期'] >= pd.to_datetime(prd_time_pair[0]))
                                      & (pick_stock_df['交易日期'] <= pd.to_datetime(prd_time_pair[1]))]
        if len(pick_stock_df_interval) == 0:
            continue

        # 用对应的训练时间构建路径
        train_time_pair = model_time_pair_dct[prd_time_pair]
        model_path = r'{}\strategy\pick_stock\model_utils\saved_model\FCN_reg_{}_{}-{}.pth'.format(project_path, train_time_pair[0], train_time_pair[1], period_type)

        model = load_model(model_path)

        # Prepare your data loader for the data you want to predict
        prediction_loader = build_prediction_data_set(feature_li, prd_time_pair[0], prd_time_pair[1], period_type, 64)

        # Make predictions
        predictions = predict(model, prediction_loader)

        pick_stock_df.loc[(pick_stock_df['交易日期'] >= pd.to_datetime(prd_time_pair[0]))
                          & (pick_stock_df['交易日期'] <= pd.to_datetime(prd_time_pair[1])), "机器学习预测值"] = predictions

    return pick_stock_df
