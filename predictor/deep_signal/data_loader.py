import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def data_split(curve_path, start_date, end_date):
    # Load dataset
    index_data = pd.read_csv(curve_path, parse_dates=['交易日期'], encoding='gbk')
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
    index_data = import_index_data(r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path), back_trader_start=date_start, back_trader_end=date_end)

    # ==============批量回测==============
    for period_type in period_type_li:
        df = pd.read_pickle(r'{}\data\historical\processed_data\all_stock_data_{}.pkl'.format(project_path, period_type))
        for strategy_name in strategy_li:
            for select_stock_num in select_stock_num_li:
                pick_time_mtd = pick_time_mtd_dct[strategy_name]
                # for pick_time_mtd in pick_time_li:
                try:
                    serial_number = generate_serial_number()
                    back_test_main(df, index_data, strategy_name, date_start, date_end, select_stock_num, period_type, serial_number,
                                   pick_time_mtd)
                except Exception as e:
                    msg = "交易播报：策略 {} 执行失败：".format(strategy_name)
                    print(msg)
                    send_dingding(msg)
                    print(e)
                    traceback.print_exc()
    X_train, y_train, X_test, y_test = data_split(curve_path, date_start, date_end)