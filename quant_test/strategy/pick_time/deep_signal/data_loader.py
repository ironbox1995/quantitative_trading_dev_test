import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def data_split(curve_path, start_date, end_date):
    # Load dataset
    data = pd.read_csv(curve_path, parse_dates=['交易日期'], encoding='gbk')
    data = data[(data['交易日期'] >= pd.to_datetime(start_date)) & (data['交易日期'] <= pd.to_datetime(end_date))]
    print("数据集长度为：{}".format(len(data)))

    # Assuming '下周期涨跌幅' is the target variable, and the rest are features
    features = data.drop(columns=['交易日期', '下周期涨跌幅'])
    targets = data['下周期涨跌幅']

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the features and transform
    scaled_features = scaler.fit_transform(features)

    # Split the dataset into training and testing sets
    # Let's use 80% of the data for training and the rest for testing
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, targets, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 两个函数输入的起止日期一定要一样，因为是在各自内部划分训练和测试数据
    strategy_name = "小市值策略"
    period_type = "W"
    select_stock_num = 3
    date_start = '2010-01-01'
    date_end = '2023-03-31'
    pick_time_mtd = "无择时"
    curve_path = r"F:\quantitative_trading_dev_test\quant_test\backtest\result_record\select_stock_{}_{}_选{}_{}-{}_{}.csv"\
        .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd)
    X_train, y_train, X_test, y_test = data_split(curve_path, date_start, date_end)