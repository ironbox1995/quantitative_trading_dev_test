from strategy.pick_time.MA_signal import *
from strategy.pick_time.LSTM_signal import *
from strategy.pick_time.index_signal import *
from utils_global.global_config import *


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
        select_stock, latest_signal = MTM_signal(select_stock, 1)
    elif pick_time_mtd == "DPO择时":
        select_stock, latest_signal = DPO_signal(select_stock, 2)
    elif pick_time_mtd == "T3择时":
        select_stock, latest_signal = T3_signal(select_stock)
    elif pick_time_mtd == "BBI择时":
        select_stock, latest_signal = BBI_signal(select_stock)
    elif pick_time_mtd == "PMO择时":
        select_stock, latest_signal = PMO_signal(select_stock)
    elif pick_time_mtd == "PO择时":
        select_stock, latest_signal = PO_signal(select_stock)
    elif pick_time_mtd == "RSIH择时":
        select_stock, latest_signal = RSIH_signal(select_stock)
    elif pick_time_mtd == "WMA择时":
        select_stock, latest_signal = WMA_signal(select_stock, 5)
    elif pick_time_mtd == "TMA择时":
        select_stock, latest_signal = TMA_signal(select_stock)
    elif pick_time_mtd == "MACD择时":
        select_stock, latest_signal = MACD_signal(select_stock)
    elif pick_time_mtd == "KDJ择时":
        select_stock, latest_signal = KDJ_signal(select_stock)
    elif pick_time_mtd == "ARRON择时":
        select_stock, latest_signal = ARRON_signal(select_stock)

    # 深度学习择时
    # elif pick_time_mtd == "LSTM择时":
    #     select_stock, latest_signal = LSTM_signal(select_stock)
    else:
        raise Exception("暂无此择时方法！")
    return select_stock, latest_signal


def curve_pick_time(select_stock, pick_time_mtd, index_data):

    if DAILY_PICK_TIME:
        daily_money_curve = cal_daily_money_curve(select_stock, index_data)
        daily_money_curve, latest_signal = pick_time(daily_money_curve, pick_time_mtd)  # 按日线数据计算信号
        select_stock = pd.merge(left=select_stock, right=daily_money_curve[['交易日期', 'signal']], on='交易日期', how='left')  # 合并到周数据
    else:
        select_stock, latest_signal = pick_time(select_stock, pick_time_mtd)

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
