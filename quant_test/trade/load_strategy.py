# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import chinese_calendar as calendar
import random

from trade.trade_config import *
from utils_global.global_config import *


def load_strategy_result(cash_amount):

    # 选股整理
    # 如果多个策略选股重复则重复买入（暂定）
    all_buy_stock = []
    for strategy_name in strategy_li:
        buy_stock_list = []
        if strategy_name != "Q学习并联策略":
            buy_stock_list.extend(split_last_line(strategy_name))
        else:
            # 尝试执行Q学习，如果报错则改为执行默认策略
            try:
                buy_stock_list.extend(load_last_line_with_biggest_q())
            except Exception as e:
                default_strategy = "小市值策略"
                buy_stock_list.extend(split_last_line(default_strategy))
                print("Q学习策略报错: {}，改为执行默认策略：{}".format(e, default_strategy))

        buy_amount = cash_amount * strategy_dct[strategy_name]

        all_buy_stock.append((buy_stock_list, buy_amount))

    return all_buy_stock


def split_last_line(strategy_name):
    # Read the CSV file
    pick_time_mtd = get_pick_time_mtd(strategy_name)

    if not Second_Board_available:
        strategy_name += "无创业"
    if not STAR_Market_available:
        strategy_name += "无科创"
    df = pd.read_csv(r"{}\backtest\latest_selection\最新选股_{}_{}_选{}_{}.csv"
            .format(project_path, strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk', parse_dates=['交易日期'])

    buy_stock_code_li = []
    signal = df['最新择时信号'].iloc[-1]

    # 判断是否为最新
    selection_day = df['交易日期'].iloc[-1]
    previous_workday = get_previous_workday()
    is_latest = (previous_workday == selection_day)
    if dev_or_test:
        is_latest = True

    if signal == 1.0 and (is_latest or force_run):
        code_column = df['买入股票代码']  # Extract the '买入股票代码' column
        last_line = code_column.iloc[-1]  # Get the last line of the column
        buy_stock_code_li = last_line.strip().split()  # Split the last line by space
    return buy_stock_code_li


def load_last_line_with_biggest_q():
    q_to_buy_stock_dct = {}
    q_to_strategy_name_dct = {}
    for strategy_name in Q_strategy_li:
        pick_time_mtd = get_pick_time_mtd(strategy_name)

        if not Second_Board_available:
            strategy_name += "无创业"
        if not STAR_Market_available:
            strategy_name += "无科创"
        df = pd.read_csv(r"{}\backtest\latest_selection\最新选股_{}_{}_选{}_{}.csv"
                .format(project_path, strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk', parse_dates=['交易日期'])
        signal = df['最新择时信号'].iloc[-1]

        # 判断是否为最新
        selection_day = df['交易日期'].iloc[-1]
        previous_workday = get_previous_workday()
        is_latest = (previous_workday == selection_day)

        if signal == 1.0 and (is_latest or force_run):
            code_column = df['买入股票代码']  # Extract the '买入股票代码' column
            last_line = code_column.iloc[-1]  # Get the last line of the column
            buy_stock_code_li = last_line.strip().split()  # Split the last line by space
            latest_q = df['Q'].iloc[-1]  # 保存q值
            q_to_buy_stock_dct[latest_q] = buy_stock_code_li
            q_to_strategy_name_dct[latest_q] = strategy_name

    # 添加空仓策略：
    q_to_buy_stock_dct[0] = []
    q_to_strategy_name_dct[0] = "空仓策略"

    random_float = random.uniform(0, 1)
    if random_float < eps:
        q_chosen = random.choice(list(q_to_buy_stock_dct.keys()))  # 从信号为1的策略中随机选择（包含空仓）
        chosen_strategy_li = q_to_buy_stock_dct[q_chosen]
        print("选取策略：", q_to_strategy_name_dct[q_chosen])
    else:
        # print(list(q_to_buy_stock_dct.keys()))
        max_q = max(list(q_to_buy_stock_dct.keys()))
        chosen_strategy_li = q_to_buy_stock_dct[max_q]
        print("选取策略：", q_to_strategy_name_dct[max_q])

    return chosen_strategy_li


def get_previous_workday():
    today = datetime.today().date()
    previous_workday = today - timedelta(days=1)

    while not calendar.is_workday(previous_workday):
        previous_workday -= timedelta(days=1)

    return previous_workday


def save_to_csv(new_row):
    pd.DataFrame(new_row, index=[0]).to_csv(r'{}\trade\交易日志.csv'.format(project_path), mode='a', header=False, index=False)


def get_pick_time_mtd(strategy_name):
    if pick_time_switch:
        return pick_time_mtd_dct[strategy_name]
    else:
        return "无择时"


if __name__ == "__main__":
    all_buy_stock = load_strategy_result(100000)
    for strategy_tup in all_buy_stock:
        print("买入列表：{}， 购买金额：{}".format(strategy_tup[0], strategy_tup[1]))
    # save_info_dct = {"日期": datetime.today().date(), "现金金额": 100000, "备注": "本周买入前金额"}
    # save_to_csv(save_info_dct)
