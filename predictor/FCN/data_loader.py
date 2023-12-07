import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from quant_test.Config.global_config import *
from predictor.FCN.filter import *


def build_stock_regression_data_set(feature_li, start_date, end_date, period_type, batch_size, data_filter):

    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)
    # 根据日期过滤数据
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df = filters(df, data_filter)  # 过滤数据
    df = df[feature_li + ["下周期涨跌幅"]]  # 只选取我们需要的特征，避免排除数据时过度排除

    df.dropna(axis=0, how="any", inplace=True)
    X = df[feature_li].values
    y = df["下周期涨跌幅"].values

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))

    # Convert the training and test sets into PyTorch datasets
    train_dataset = TensorDataset(X_train.float(), y_train.float())
    test_dataset = TensorDataset(X_test.float(), y_test.float())

    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def build_prediction_data_set(feature_li, start_date, end_date, period_type, batch_size, data_filter):
    # 加载数据
    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)
    # 根据日期过滤数据
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df = filters(df, data_filter)  # 过滤数据
    df = df[feature_li]  # 只选取我们需要的特征，避免排除数据时过度排除

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
