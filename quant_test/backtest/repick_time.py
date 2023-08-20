from strategy.pick_time.MA_signal import *
from strategy.pick_time.LSTM_signal import *
from strategy.pick_time.index_signal import *


def pick_time(select_stock, pick_time_mtd):

    # 均线择时
    if pick_time_mtd == "双均线择时":
        select_stock, latest_signal = MA_signal(select_stock, para=(1, 3))

    # 指标择时
    elif pick_time_mtd == "MICD择时":
        select_stock, latest_signal = MICD_signal(select_stock)
    elif pick_time_mtd == "SROC择时":
        select_stock, latest_signal = SROC_signal(select_stock)
    elif pick_time_mtd == "ENV择时":
        select_stock, latest_signal = ENV_signal(select_stock)
    elif pick_time_mtd == "MTM择时":
        select_stock, latest_signal = MTM_signal(select_stock)
    elif pick_time_mtd == "DPO择时":
        select_stock, latest_signal = DPO_signal(select_stock)

    # 深度学习择时
    # elif pick_time_mtd == "LSTM择时":
    #     select_stock, latest_signal = LSTM_signal(select_stock)
    else:
        raise Exception("暂无此择时方法！")
    return select_stock, latest_signal


def curve_pick_time(select_stock, pick_time_mtd):
    select_stock, latest_signal = pick_time(select_stock, pick_time_mtd)

    select_stock['资金曲线_择时'] = (select_stock['选股下周期涨跌幅'] * select_stock['signal'] + 1).cumprod()  # 计算资金曲线

    # 将信号为0的涨跌幅置为0
    select_stock['选股下周期每天涨跌幅'] = select_stock.apply(lambda row: [0.0] * len(row['选股下周期每天涨跌幅']) if row['signal'] == 0 else row['选股下周期每天涨跌幅'], axis=1)
    select_stock['选股下周期涨跌幅'] = select_stock.apply(lambda row: 0.0 if row['signal'] == 0 else row['选股下周期涨跌幅'], axis=1)

    # 将股票数量置为0
    select_stock['股票数量'] = select_stock.apply(lambda row: 0.0 if row['signal'] == 0 else row['股票数量'], axis=1)

    # 将买入股票代码和买入股票名称置为空
    select_stock.loc[select_stock['signal'] == 0, ['买入股票代码', '买入股票名称']] = 'empty'

    return select_stock

