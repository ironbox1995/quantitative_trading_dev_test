# 特征工程

# 全部数据列：	交易日期	股票代码	股票名称	是否交易	开盘价	最高价	最低价	收盘价	成交额	流通市值（万元）	总市值 （万元）	上市至今交易天数	下日_是否交易	下日_开盘涨停	下日_是否ST	下日_是否S
# 下日_是否退市	下日_开盘买入涨跌幅	行业	市场类型	成交量	量比	市盈率	市盈率TTM	市净率	市销率	市销率TTM
# 股息率（%）	股息率TTM（%）	单日振幅20日均值	20日振幅	VWAP	换手率（%）	当日换手率	5日均线	20日均线	bias_5	5日涨跌幅环比变化
# bias_20	20日涨跌幅环比变化	量价相关性	周期内成交额	周期内最后交易日流通市值	周期换手率	每天涨跌幅	本周期涨跌幅	本周期指数涨跌幅	下周期每天涨跌幅	下周期涨跌幅

# 可用特征： 开盘价、最高价、最低价、收盘价、成交额、流通市值（万元）、总市值 （万元）、成交量、量比、市盈率、市盈率TTM、市净率、市销率、市销率TTM、股息率（%）、股息率TTM（%）、20日振幅、VWAP、换手率（%）、5日均线、20日均线、bias_5、bias_20、本周期涨跌幅

import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
# from mlxtend.feature_selection import SequentialFeatureSelector

from machine_learning.data_loader import status_calc
from machine_learning.model_config import *

# feature_li = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '流通市值（万元）', '总市值 （万元）', '成交量', '量比', '市盈率', '市盈率TTM', '市净率', '市销率', '市销率TTM', '股息率（%）', '股息率TTM（%）', '20日振幅', 'VWAP', '换手率（%）', '5日均线', '20日均线', 'bias_5', 'bias_20', '本周期涨跌幅']
data_li = ["本周期涨跌幅", "本周期指数涨跌幅", "下周期涨跌幅"]

feature_li = ['开盘价', '最高价', '最低价', '收盘价', '成交额', '流通市值（万元）', '总市值 （万元）', '成交量', '量比', '市盈率', '市盈率TTM', '市净率', '市销率', '市销率TTM', '股息率（%）', '股息率TTM（%）', '20日振幅', 'VWAP', '换手率（%）', '5日均线', '20日均线', 'bias_5', 'bias_20', '本周期涨跌幅']


def feature_for_SVC(df_train, df_data):
    kernel = 'rbf'
    X = df_train.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = [status_calc(df_data["下周期涨跌幅"].values[i],  # 应该预测下周期的涨跌幅
                           0,
                           out_performance) for i in range(len(df_train))]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform feature selection using Sequential Feature Selector
    svc = SVC(kernel=kernel)
    sfs = SequentialFeatureSelector(svc, k_features='best', forward=True, floating=False, verbose=1, scoring='f1')
    X_train_selected = sfs.fit_transform(X_train_scaled, y_train)
    X_test_selected = sfs.transform(X_test_scaled)

    # Train the SVC model on the transformed features
    svc_model = SVC(kernel=kernel)
    svc_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svc_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Train the SVC model on the transformed features
    svc_model_selected = SVC(kernel=kernel)
    svc_model_selected.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred_selected = svc_model_selected.predict(X_test_selected)
    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected)

    print("选择特征的结果为：{}".format(list(sfs.k_feature_idx_)))
    X_df = pd.DataFrame(X_train, columns=feature_li)
    X_SFS1 = X_df.iloc[:, list(sfs.k_feature_idx_)]
    with open('feature_selection_result.txt', 'a') as f:
        print("="*30, file=f)
        print("执行时间：{}".format(datetime.datetime.now()), file=f)
        print("数据总量：{}".format(len(df_data)), file=f)
        print("正例比例：{}".format(sum(y)/len(y)), file=f)
        print("预测结果中的正例比例：{}".format(sum(y_pred_selected)/len(y_pred_selected)), file=f)
        print("SVC前向选择特征的结果为：{}，index：{}".format(X_SFS1.columns, list(sfs.k_feature_idx_)), file=f)
        print("SFS前向选择前的结果: acc: {}, f1: {}".format(accuracy, f1), file=f)
        print("SFS前向选择后的结果: acc: {}, f1: {}".format(accuracy_selected, f1_selected), file=f)


if __name__ == "__main__":
    data_type = "W"
    train_start_date = "2022-12-01"
    train_end_date = "2022-12-31"

    data_path = r"F:\quantitative_trading\quant_formal\data\historical\processed_data\all_stock_data_{}.pkl".format(
        data_type)
    df = pd.read_pickle(data_path)
    df = df[(df['交易日期'] >= pd.to_datetime(train_start_date)) & (df['交易日期'] <= pd.to_datetime(train_end_date))]
    df.dropna(axis=0, how="any", inplace=True)
    df_train = df[feature_li]
    df_data = df[data_li]

    # y_train = [status_calc(df_data["下周期涨跌幅"].values[i],  # 应该预测下周期的涨跌幅
    #                        0,
    #                        0.1) for i in range(len(df_data))]

    # print("数据总量：{}".format(len(df_data)))
    # print("正例比例:{}".format(len(y_train)/sum(y_train)))

    print("开始对SVC进行特征工程：时间：{}".format(datetime.datetime.now()))
    feature_for_SVC(df_train, df_data)
    print("SVC特征工程结束：时间：{}".format(datetime.datetime.now()))

