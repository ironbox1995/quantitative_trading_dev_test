# -*- coding: utf-8 -*-
import ast

from data.processing.Functions import *
from MA_signal import *
from LSTM_signal import *


def pick_time(select_stock, pick_time_mtd, para=(1, 3)):
    if pick_time_mtd == "双均线择时":
        select_stock, latest_signal = MA_signal(select_stock, para)
    elif pick_time_mtd == "LSTM择时":
        select_stock, latest_signal = LSTM_signal(select_stock)
    else:
        raise Exception("暂无此择时方法！")
    return select_stock, latest_signal


def curve_pick_time(select_stock, pick_time_mtd, para=(1, 3)):
    select_stock, latest_signal = pick_time(select_stock, pick_time_mtd, para)

    select_stock['资金曲线_择时'] = (select_stock['选股下周期涨跌幅'] * select_stock['signal'] + 1).cumprod()  # 计算资金曲线

    # 将信号为0的涨跌幅置为0
    select_stock['选股下周期每天涨跌幅'] = select_stock.apply(lambda row: [0.0] * len(row['选股下周期每天涨跌幅']) if row['signal'] == 0 else row['选股下周期每天涨跌幅'], axis=1)
    select_stock['选股下周期涨跌幅'] = select_stock.apply(lambda row: 0.0 if row['signal'] == 0 else row['选股下周期涨跌幅'], axis=1)

    # 将股票数量置为0
    select_stock['股票数量'] = select_stock.apply(lambda row: 0.0 if row['signal'] == 0 else row['股票数量'], axis=1)

    # 将买入股票代码和买入股票名称置为空
    select_stock.loc[select_stock['signal'] == 0, ['买入股票代码', '买入股票名称']] = 'empty'

    return select_stock


def collect_column_data(df, column_name):
    # TODO: 仍然不能正确处理
    column_data = df[column_name].tolist()
    combined_data = []

    for row in column_data:
        print(type(row))
        print(row)
        combined_data.extend(row)

    return combined_data


def find_best_pick_time_para(para=(1, 3)):

    strategy_name = "小市值策略"
    period_type = 'W'
    select_stock_num = 3
    date_start = '2010-01-01'
    date_end = '2023-07-07'
    pick_time_mtd = "双均线择时"

    # 导入指数数据
    index_data = import_index_data(
        r"F:\quantitative_trading\quant_formal\data\historical\tushare_index_data\000001.SH.csv"
        , back_trader_start=date_start, back_trader_end=date_end)

    select_stock = pd.read_csv(
        r"F:\quantitative_trading\quant_formal\backtest\result_record\select_stock_{}_{}_选{}_{}-{}_{}.csv"
            .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd),
        encoding='gbk', parse_dates=['交易日期'])

    # 根据资金曲线择时
    if pick_time_mtd == "" or pick_time_mtd == "无择时":
        pick_time_mtd = "无择时"
    else:
        select_stock = curve_pick_time(select_stock, pick_time_mtd, para)


    # ===计算选中股票每天的资金曲线
    # 计算每日资金曲线
    equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                      how='left', sort=True)  # 将选股结果和大盘指数合并

    equity['持有股票代码'] = equity['买入股票代码'].shift()
    equity['持有股票代码'].fillna(method='ffill', inplace=True)
    equity.dropna(subset=['持有股票代码'], inplace=True)
    # del equity['买入股票代码']

    combined_data = collect_column_data(select_stock, '选股下周期每天涨跌幅')
    equity['涨跌幅'] = combined_data
    equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
    equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()

    # equity.to_csv(r"F:\quantitative_trading\quant_formal\backtest\result_record\equity_{}_{}_选{}_{}-{}_{}.csv"
    #               .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd),
    #               encoding='gbk')

    last_equity_value = equity.tail(1)['equity_curve'].iloc[0]
    # print(last_equity_value)

    return last_equity_value


if __name__ == "__main__":
    # short_list = [1, 1, 1, 1, 2, 2, 3, 3, 5, 5, 10, 10]
    # long_list = [2, 3, 4, 5, 3, 5, 5, 6, 10, 20, 20, 50]
    #
    # for (short_avg_length, long_avg_length) in zip(short_list, long_list):
    #     last_equity_value = find_best_pick_time_para((short_avg_length, long_avg_length))
    #     print(last_equity_value)

    max_last_equity_value = 0
    best_long = 0
    best_short = 0

    print("开始执行搜索：")
    for long in range(2, 120):
        print(long)
        for short in range(1, long):
            last_equity_value = find_best_pick_time_para((short, long))
            if last_equity_value > max_last_equity_value:
                max_last_equity_value = last_equity_value
                best_long = long
                best_short = short

    print("最佳净值：{}，最佳短均线：{}，最佳长均线：{}".format(max_last_equity_value, best_short, best_long))

