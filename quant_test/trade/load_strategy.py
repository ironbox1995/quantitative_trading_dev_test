# -*- coding: utf-8 -*-
import pandas as pd
from datetime import timedelta
import chinese_calendar as calendar
import random
import warnings

from Config.trade_config import *
from Config.global_config import *
from utils_global.dingding_message import *

warnings.filterwarnings('ignore')


def load_strategy_result():

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

        fund_share = strategy_part_dct[strategy_name]
        all_buy_stock.append((buy_stock_list, fund_share))

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
    df_draw_down = pd.read_csv(r"{}\backtest\latest_selection\最近回撤_{}_{}_选{}_{}.csv"
            .format(project_path, strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk', parse_dates=['交易日期'])
    last_draw_down = df_draw_down['最近回撤幅度'].iloc[-1]

    buy_stock_code_li = []

    # 判断择时信号
    signal = df['最新择时信号'].iloc[-1]

    # 判断是否为最新结果
    selection_day = df['交易日期'].iloc[-1]
    previous_workday = get_previous_workday()
    is_latest = (previous_workday == selection_day)
    if dev_or_test:
        is_latest = True

    # 判断回撤幅度是否需要告警或止损
    full_name = "{}_{}_选{}_{}".format(strategy_name, period_type, select_stock_num, pick_time_mtd)
    preset_draw_down = strategy_stop_loss_point_dct[full_name]
    if preset_draw_down > 0:  # 回撤非正，所以如果设为正数则不止损
        strategy_loss_permission = True
    else:
        if last_draw_down < preset_draw_down * draw_down_warning_point:  # 回撤大于特定比例则告警
            send_dingding(f"交易播报：策略：{strategy_name} 最近回撤为 {last_draw_down * 100}%，"
                          f"大于预设值 {preset_draw_down * 100}% 的 {draw_down_warning_point * 100}%，"
                          f"告警！需引起重视！")
            strategy_loss_permission = True
        elif last_draw_down < preset_draw_down:  # 回撤大于特定比例则止损
            strategy_loss_permission = False
        else:  # 没有此类情况则正常执行
            strategy_loss_permission = True

    if signal == 1.0 and (is_latest or force_run) and strategy_loss_permission:
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
    try:
        pd.DataFrame(new_row, index=[0]).to_csv(r'{}\trade\交易日志.csv'.format(project_path), mode='a', header=False, index=False)
    except:
        print("交易日志保存失败，待保存数据为：")
        print(new_row)


def get_pick_time_mtd(strategy_name):
    if pick_time_switch:
        return pick_time_mtd_dct[strategy_name]
    else:
        return "无择时"


if __name__ == "__main__":
    all_buy_stock = load_strategy_result()
    for strategy_tup in all_buy_stock:
        print("买入列表：{}， 购买比例：{}".format(strategy_tup[0], strategy_tup[1]))
