import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib  # 用于保存和加载规范化器

from quant_test.Config.global_config import *
from predictor.FCN.filter import *


def normalize_data(X_train, X_test, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid normalization method. Choose 'standard', 'minmax', or 'robust'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled


def build_stock_regression_data_set(feature_li, start_date, end_date, period_type, batch_size, data_filter
                                    , norm_method='standard'):
    # 加载数据
    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)

    # 根据日期过滤数据
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df = filters(df, data_filter)  # 过滤数据
    df = df[feature_li + ["下周期涨跌幅"]]  # 只选取我们需要的特征

    # 删除缺失值
    df.dropna(axis=0, how="any", inplace=True)
    X = df[feature_li].values
    y = df["下周期涨跌幅"].values

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 规范化数据
    scaler, X_train_scaled, X_test_scaled = normalize_data(X_train, X_test, method=norm_method)

    # 如果指定了路径，保存规范化器
    scaler_save_path = f'scaler_FCN_reg_{start_date}_{end_date}-{period_type}-{data_filter}'
    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)

    # 转换为PyTorch张量
    X_train, X_test, y_train, y_test = map(torch.tensor, (X_train_scaled, X_test_scaled, y_train, y_test))

    # 创建PyTorch数据集
    train_dataset = TensorDataset(X_train.float(), y_train.float())
    test_dataset = TensorDataset(X_test.float(), y_test.float())

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def build_prediction_data_set(feature_li, start_date, end_date, period_type, batch_size, data_filter):
    # 加载数据
    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)

    # 根据日期过滤数据
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df = filters(df, data_filter)  # 过滤数据
    df = df[feature_li]  # 只选取我们需要的特征

    # 删除缺失值
    df.dropna(axis=0, how="any", inplace=True)
    X = df[feature_li].values

    # 加载保存的规范化器
    scaler_load_path = f'scaler_FCN_reg_{start_date}_{end_date}-{period_type}-{data_filter}'
    scaler = joblib.load(scaler_load_path)

    # 规范化数据
    X_normalized = scaler.transform(X)

    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_normalized).float()

    # 创建PyTorch数据集
    prediction_dataset = TensorDataset(X_tensor)

    # 创建数据加载器
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)

    return prediction_loader
