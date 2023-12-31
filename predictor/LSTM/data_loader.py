import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 用于保存和加载规范化器

from quant_test.Config.global_config import *
from predictor.LSTM.filter import *


def normalize_dataframe(df, feature_li, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[feature_li] = scaler.fit_transform(df[feature_li])
    else:
        df[feature_li] = scaler.transform(df[feature_li])
    return df, scaler


def generate_sequences(df, feature_li, target_col, sequence_length):
    sequences = []
    for i in range(len(df) - sequence_length):
        if target_col:
            sequence = df[feature_li + [target_col]].iloc[i:i + sequence_length]
        else:
            sequence = df[feature_li].iloc[i:i + sequence_length]
        sequences.append(sequence.values)
    return np.array(sequences)


def build_lstm_stock_regression_data_set(feature_li, start_date, end_date, period_type, batch_size, data_filter
                                         , sequence_length=20):
    # 加载数据
    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df = filters(df, data_filter)  # 过滤数据
    df = df[feature_li + ["下周期涨跌幅"]]  # 选择特定特征和目标列
    df.dropna(axis=0, how="any", inplace=True)

    # 规范化特征
    scaler_save_path = f'scaler_LSTM_reg_{start_date}_{end_date}-{period_type}-{data_filter}'
    df, scaler = normalize_dataframe(df, feature_li)
    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)

    # 生成序列数据（包含目标列）
    sequences = generate_sequences(df, feature_li, "下周期涨跌幅", sequence_length)
    X = sequences[:, :, :-1]
    y = sequences[:, -1, -1]

    # 数据拆分与转换
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))

    # 创建数据集和加载器
    train_dataset = TensorDataset(X_train.float(), y_train.float())
    test_dataset = TensorDataset(X_test.float(), y_test.float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def build_lstm_prediction_data_set(feature_li, start_date, end_date, period_type, batch_size, data_filter
                                   , sequence_length=20):
    # 加载数据
    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, period_type)
    df = pd.read_pickle(data_path)
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df = filters(df, data_filter)  # 过滤数据
    df = df[feature_li]  # 仅选择特定特征
    df.dropna(axis=0, how="any", inplace=True)

    # 加载保存的规范化器
    scaler_load_path = f'scaler_LSTM_reg_{start_date}_{end_date}-{period_type}-{data_filter}'
    scaler = joblib.load(scaler_load_path)
    df, _ = normalize_dataframe(df, feature_li, scaler)

    # 生成序列数据（仅特征）
    sequences = generate_sequences(df, feature_li, None, sequence_length)
    X = sequences

    # 数据转换
    X_tensor = torch.tensor(X).float()

    # 创建数据集和加载器
    prediction_dataset = TensorDataset(X_tensor)
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)

    return prediction_loader
