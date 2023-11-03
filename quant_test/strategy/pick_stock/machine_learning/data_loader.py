from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from strategy.pick_stock.machine_learning.model_config import *
from Config.global_config import *


def status_calc(stock, beta_increase, out_performance=0.1):
    """A simple function to classify whether a stock outperformed the S&P500
    :param stock: stock price
    :param beta_increase:
    :param out_performance: stock is classified 1 if stock price > S&P500 price + outperformance
    :return: true/false
    """
    if out_performance < 0:
        raise ValueError("outperformance must be positive")
    return 1 if (stock - beta_increase >= out_performance) and stock > 0.002 else 0


def build_classify_data_set(start_date, end_date, data_type):

    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, data_type)
    df = pd.read_pickle(data_path)
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]

    df.dropna(axis=0, how="any", inplace=True)
    X = df[feature_li].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    y = np.array([status_calc(df["下周期涨跌幅"].values[i],
                     0,
                     out_performance) for i in range(len(X))])
    # Normalize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split X and y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=SHUFFLE)

    return X_train, y_train, X_test, y_test


def build_regression_data_set(start_date, end_date, data_type):

    data_path = r"{}\data\historical\processed_data\all_stock_data_{}.pkl".format(project_path, data_type)
    df = pd.read_pickle(data_path)
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]

    df.dropna(axis=0, how="any", inplace=True)
    X = df[feature_li].values
    y = df["下周期涨跌幅"].values

    # Normalize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split X and y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=SHUFFLE)

    return X_train, y_train, X_test, y_test


def build_classify_prediction_data_set(training_data):
    training_data.dropna(axis=0, how="any", inplace=True)
    X_train = training_data[feature_li].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled


def build_regression_prediction_data_set(training_data):
    training_data.dropna(axis=0, how="any", inplace=True)
    X_train = training_data[feature_li].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled


# Define a function to create the validation loader
def create_rgs_val_loader(X_val, y_val):
    val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    val_loader = DataLoader(val_data, batch_size=fcn_batch_size, shuffle=False)
    return val_loader


# Define a function to create the validation loader
def create_clf_val_loader(X_val, y_val):
    val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    val_loader = DataLoader(val_data, batch_size=fcn_batch_size, shuffle=False)
    return val_loader


if __name__ == "__main__":
    data_type = "M"

    for time_pair in model_time_pair_tpl:
        X_train, y_train, X_test, y_test =build_regression_data_set(time_pair[0], time_pair[1], data_type)
        # print(X_train[:5])
        # print(y_train[:5])
