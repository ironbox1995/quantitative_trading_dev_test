# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import chinese_calendar as calendar

from trade.trade_config import *


def load_strategy_result(cash_amount):

    # 选股整理
    # 如果多个策略选股重复则只选一次(暂定)
    buy_stock_list = []
    for strategy_name in strategy_li:
        buy_stock_list.extend(split_last_line(strategy_name))

    # TODO: 仓位管理
    buy_amount = cash_amount

    return buy_stock_list, buy_amount


def split_last_line(strategy_name):
    # Read the CSV file
    df = pd.read_csv(r"F:\quantitative_trading_dev_test\quant_test\backtest\latest_selection\最新选股_{}_{}_选{}_{}.csv"
            .format(strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk', parse_dates=['交易日期'])

    buy_stock_code_li = []
    signal = df['最新择时信号'].iloc[-1]

    # 判断是否为最新
    selection_day = df['交易日期'].iloc[-1]
    previous_workday = get_previous_workday()
    is_latest = (previous_workday == selection_day)

    if signal == 1.0 and is_latest:
        code_column = df['买入股票代码']  # Extract the '买入股票代码' column
        last_line = code_column.iloc[-1]  # Get the last line of the column
        buy_stock_code_li = last_line.strip().split()  # Split the last line by space
    return buy_stock_code_li


def get_previous_workday():
    today = datetime.today().date()
    previous_workday = today - timedelta(days=1)

    while not calendar.is_workday(previous_workday):
        previous_workday -= timedelta(days=1)

    return previous_workday


if __name__ == "__main__":
    buy_stock_list, buy_amount = load_strategy_result(100000)
    print("买入列表：{}， 购买金额：{}".format(buy_stock_list, buy_amount))
