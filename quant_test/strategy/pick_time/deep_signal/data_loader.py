import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from deep_signal.pick_time_utils import *


def data_split(curve_path, start_date, end_date, CLASSIFY=False):
    # Load dataset
    dataset = pd.read_csv(curve_path, parse_dates=['交易日期'], encoding='gbk')
    dataset = dataset[(dataset['交易日期'] >= pd.to_datetime(start_date)) & (dataset['交易日期'] <= pd.to_datetime(end_date))]
    print("数据集长度为：{}".format(len(dataset)))

    # Normalize the data
    fund_curve = dataset['资金曲线'].values
    # fund_curve_5d_avg = np.convolve(fund_curve, np.ones((5,)) / 5, mode='valid')
    # fund_curve_5d_avg = np.concatenate((np.zeros(4), fund_curve_5d_avg))
    # fund_curve_10d_avg = np.convolve(fund_curve, np.ones((10,)) / 10, mode='valid')
    # fund_curve_10d_avg = np.concatenate((np.zeros(9), fund_curve_10d_avg))

    # Define features and target
    if DIFF or CLASSIFY:
        increase_rate = [(fund_curve[i] - fund_curve[i - 1])/fund_curve[i - 1] for i in range(1, len(fund_curve))]
        increase_rate_std = [(increase_rate[i] - np.mean(increase_rate)) / np.std(increase_rate) for i in range(1, len(increase_rate))]
        X = np.array([increase_rate_std[i - PRCT_LENGTH:i] for i in range(PRCT_LENGTH, len(increase_rate_std))])
        y = np.array([increase_rate_std[i] for i in range(PRCT_LENGTH, len(increase_rate_std))])
    else:
        fund_curve = (fund_curve - np.mean(fund_curve)) / np.std(fund_curve)
        X = np.array([fund_curve[i - PRCT_LENGTH:i] for i in range(PRCT_LENGTH, len(fund_curve))])
        y = np.array([fund_curve[i] for i in range(PRCT_LENGTH, len(fund_curve))])
    if CLASSIFY:
        y = np.array([1 if i > 0 else 0 for i in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=SHUFFLE)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 两个函数输入的起止日期一定要一样，因为是在各自内部划分训练和测试数据
    strategy_name = "小市值策略"
    period_type = "W"
    select_stock_num = 3
    date_start = '2010-01-01'
    date_end = '2023-03-31'
    pick_time_mtd = "无择时"
    curve_path = r"F:\quantitative_trading\quant_formal\backtest\result_record\select_stock_{}_{}_选{}_{}-{}_{}.csv"\
        .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd)
    X_train, y_train, X_test, y_test = data_split(curve_path, date_start, date_end)