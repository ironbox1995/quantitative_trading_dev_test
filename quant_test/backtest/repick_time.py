import pandas as pd
from Config.global_config import *
from strategy.get_strategy_function import get_pick_time_strategy


def curve_pick_time(select_stock, pick_time_mtd, index_data):

    if DAILY_PICK_TIME:
        daily_money_curve = cal_daily_money_curve(select_stock, index_data)
        daily_money_curve, latest_signal = get_pick_time_strategy(daily_money_curve, pick_time_mtd)  # 按日线数据计算信号
        select_stock = pd.merge(left=select_stock, right=daily_money_curve[['交易日期', 'signal']], on='交易日期', how='left')  # 合并到周数据
    else:
        select_stock, latest_signal = get_pick_time_strategy(select_stock, pick_time_mtd)

    select_stock['资金曲线_择时'] = (select_stock['选股下周期涨跌幅'] * select_stock['signal'] + 1).cumprod()  # 计算资金曲线

    # 将信号为0的涨跌幅置为0
    select_stock['选股下周期每天涨跌幅'] = select_stock.apply(lambda row: [0.0] * len(row['选股下周期每天涨跌幅']) if row['signal'] == 0 else row['选股下周期每天涨跌幅'], axis=1)
    select_stock['选股下周期涨跌幅'] = select_stock.apply(lambda row: 0.0 if row['signal'] == 0 else row['选股下周期涨跌幅'], axis=1)

    # 将股票数量置为0
    select_stock['股票数量'] = select_stock.apply(lambda row: 0.0 if row['signal'] == 0 else row['股票数量'], axis=1)

    # 将买入股票代码和买入股票名称置为空
    select_stock.loc[select_stock['signal'] == 0, ['买入股票代码', '买入股票名称']] = 'empty'

    return select_stock, latest_signal


def cal_daily_money_curve(select_stock, index_data):
    # ===计算选中股票每天的资金曲线
    # 计算每日资金曲线
    daily_money_curve = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                      how='left', sort=True)  # 将选股结果和大盘指数合并

    daily_money_curve['持有股票代码'] = daily_money_curve['买入股票代码'].shift()
    daily_money_curve['持有股票代码'].fillna(method='ffill', inplace=True)
    daily_money_curve.dropna(subset=['持有股票代码'], inplace=True)
    del daily_money_curve['买入股票代码']

    daily_money_curve['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()
    daily_money_curve['资金曲线'] = (daily_money_curve['涨跌幅'] + 1).cumprod()

    return daily_money_curve
