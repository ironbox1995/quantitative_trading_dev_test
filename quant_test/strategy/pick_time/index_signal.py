import pandas as pd
import numpy as np


# =========MICD 异同离差动力指数========
# MICD策略
def MICD_signal(select_stock, para=(2, 4)):
    """
    https://bbs.quantclass.cn/thread/5630
    MICD 异同离差动力指数
    如果 MICD 上穿 0，则产生买入信号；
    如果 MICD 下穿 0，则产生卖出信号。
    :param select_stock:
    :param para: N, M
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    MICD_N = para[0]  # 短期均线。ma代表：moving_average
    MICD_M = para[1]  # 长期均线

    N1 = 2
    N2 = 4

    # ===计算均线。所有的指标，都要使用复权价格进行计算。
    select_stock['MICD_MI'] = select_stock['资金曲线'] - select_stock['资金曲线'].shift()
    select_stock['MICD_MTMMA'] = SMA_CN(select_stock['MICD_MI'].tolist(), MICD_N)
    select_stock['MICD_MTMMA1'] = select_stock['MICD_MTMMA'].shift()
    select_stock['MICD_DIF'] = select_stock['MICD_MTMMA1'].rolling(window=N1).mean() - select_stock[
        'MICD_MTMMA1'].rolling(window=N2).mean()
    select_stock['MICD'] = SMA_CN(select_stock['MICD_DIF'].tolist(), MICD_M)
    del select_stock['MICD_MI'], select_stock['MICD_MTMMA'], select_stock['MICD_MTMMA1'], select_stock['MICD_DIF']

    # ===找出做多信号
    condition1 = select_stock['MICD'] > 0  # MICD > 0
    condition2 = select_stock['MICD'].shift(1) < 0  # MICD < 0
    select_stock.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出平仓信号
    condition1 = select_stock['MICD'] < 0  # MICD < 0
    condition2 = select_stock['MICD'].shift(1) > 0  # MICD > 0
    select_stock.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号的那根K线的signal设置为0，0代表平仓

    select_stock['signal'].fillna(method='ffill', inplace=True)
    latest_signal = select_stock.tail(1)['signal'].iloc[0]
    select_stock['signal'] = select_stock['signal'].shift(1)  # 产生的信号下个周期才能用
    select_stock['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可
    # select_stock['signal'].fillna(method='ffill', inplace=True)

    # ===删除无关中间变量
    select_stock.drop(['MICD'], axis=1, inplace=True)

    return select_stock, latest_signal


# 手工计算SMA，talib的不对
def SMA_CN(arr, n, m=1):
    n = int(n)
    m = int(m)
    y = 0
    result = []

    for x in arr:
        if np.isnan(x):
            x = np.nan_to_num(x)
        y = (m * x + (n - m) * y) / n
        result.append(y)

    return pd.Series(result)


# SROC策略
def SROC_signal(df, para=(2, 3, 5)):
    """
    https://bbs.quantclass.cn/thread/6004
    SROC衡量价格涨跌幅，反映市场的超买或超卖状态，属于价格反转因子
    1)当SROC过高时，市场处于超买状态；
    2)当SROC过低时，市场处于超卖状态。

    :param df:
    :param para:
        N: 周期1
        M: 周期2
        rt: 超买或超卖线的阈值
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    N = para[0]  # N个周期的移动平均
    M = para[1]  # M个周期的SROC
    rt = para[2] / 100  # 信号阈值

    # ===计算SROC。所有的指标，都要使用复权价格进行计算。
    df['EMAP'] = df['资金曲线'].rolling(N).mean()
    df['sroc'] = df['EMAP'] / df['EMAP'].shift(M) - 1
    # ===找出做多信号
    # condition1 = df['sroc'] < -rt  # SROC 下穿阈值做多
    condition1 = df['sroc'] > rt
    df.loc[condition1, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    # condition1 = df['roc'] > rt
    condition1 = df['sroc'] < -rt  # SROC 下穿阈值做多
    df.loc[condition1, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['sroc'], axis=1, inplace=True)

    return df, latest_signal


# ENV策略
def ENV_signal(df, para=(2, 5)):
    """
    https://bbs.quantclass.cn/thread/6151
    ENV(Envolope 包络线)指标是由移动平均线上下平移一定的幅度 (百分比)所得。
    当价格突破上轨时再产生 买入信号或者当价格突破下轨再产生卖出信号。
    :param df:
    :param para: N, PARAM
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """
    N = para[0]
    PARAM = para[1] / 100

    df['MAC'] = df['资金曲线'].rolling(N, min_periods=1).mean()
    df['UPPER'] = df['MAC'] * (1 + PARAM)
    df['LOWER'] = df['MAC'] * (1 - PARAM)

    # ===找出做多信号
    condition1 = df['资金曲线'] > df['UPPER']  # 资金曲线 > 上轨线
    condition2 = df['资金曲线'].shift(1) <= df['UPPER'].shift(1)  # 上一周期的资金曲线 <= 上轨线
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['资金曲线'] < df['LOWER']  # 资金曲线 < 下轨线
    condition2 = df['资金曲线'].shift(1) >= df['LOWER'].shift(1)  # 上一周期的资金曲线 >= 下轨线
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['MAC', 'UPPER', 'LOWER'], axis=1, inplace=True)

    return df, latest_signal


# MTM策略
def MTM_signal(df, para=1):
    """
    https://bbs.quantclass.cn/thread/19630
    N=60
    MTM=CLOSE-REF(CLOSE,N)
    MTM 用当天价格与 N 天前价格的差值来衡量价格的动量。如果 MTM
    上穿/下穿 0 则产生买入/卖出信号。

    """
    # print(para)
    # ===策略参数
    df['MTM'] = df['资金曲线'] - df['资金曲线'].shift(para)

    # ===找出做多信号
    condition1 = df['MTM'] >= 0
    df.loc[condition1, 'signal'] = 1

    # ===找出做多平仓信号
    condition1 = df['MTM'] < 0
    df.loc[condition1, 'signal'] = 0

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['MTM'], axis=1, inplace=True)
    return df, latest_signal


# =====DPO指标再择时
# DPO指标
def DPO_signal(df, para=2):
    """
    DPO=CLOSE-REF(MA(CLOSE,N),N/2+1),资金曲线 - N/2+1天前的 N周期均线
    DPO是当前价格与延迟的移动平均线的差值，通过去除前一段时间的移动平均价格来减少长期的趋势对短期价格波动的影响。
    DPO>0，表明目前处于多头市场；DPO<0，表明目前处于空头市场。
    如果 DPO上穿0线，做多; 如果DPO下穿0线，平仓。
    :param df:
    :param para:
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """
    # ===策略参数
    n = para

    # ===计算n周期ma均线
    df['N周期均线'] = df['资金曲线'].rolling(n, min_periods=1).mean()

    # ===计算DPO指标。所有的指标都要使用复权价格进行计算。
    # 根据DPO=CLOSE-REF(MA(CLOSE,N),N/2+1),计算DPO
    df['DPO'] = df['资金曲线'] - df['N周期均线'].shift(int(n / 2 + 1))

    # ===找出做多信号
    condition1 = df['DPO'] > 0  # 本周期DPO > 0
    condition2 = df['DPO'].shift(1) <= 0  # 上一周期的DPO <= 0
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['DPO'] < 0  # 本周期DPO < 0
    condition2 = df['DPO'].shift(1) >= 0  # 上一周期的DPO >= 0
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['N周期均线'], axis=1, inplace=True)

    return df, latest_signal


def T3_signal(df, para=(2, 1)):
    """
    https://bbs.quantclass.cn/thread/31096
    :param df:
    :param para:
    :return:
    """
    N = int(para[0])
    VA = float(para[1])

    df['EMA'] = df['资金曲线'].ewm(span=N, adjust=False).mean()
    df['T1'] = df['EMA'] * (1 + VA) - df['EMA'].ewm(span=N, adjust=False).mean() * VA
    df['EMA_T1'] = df['T1'].ewm(span=N, adjust=False).mean()
    df['T2'] = df['EMA_T1'] * (1 + VA) - df['EMA_T1'].ewm(span=N, adjust=False).mean() * VA
    df['EMA_T2'] = df['T2'].ewm(span=N, adjust=False).mean()
    df['T3'] = df['EMA_T2'] * (1 + VA) - df['EMA_T2'].ewm(span=N, adjust=False).mean() * VA

    # ===计算信号代码
    condition = df['资金曲线'] > df['T3']
    condition &= df['资金曲线'].shift() <= df['T3'].shift()
    df.loc[condition, 'signal'] = 1

    condition = df['资金曲线'] < df['T3']
    condition &= df['资金曲线'].shift() >= df['T3'].shift()
    df.loc[condition, 'signal'] = 0

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['EMA', 'T1', 'EMA_T1', 'T2', 'EMA_T2', 'T3'], axis=1, inplace=True)

    return df, latest_signal


# BBI策略
def BBI_signal(df, para=(2, 3, 5, 13)):
    """
    https://bbs.quantclass.cn/thread/30938
    BBI 是对不同时间长度的移动平均线取平均，能够综合不同移动平均
    线的平滑性和滞后性。如果资金曲线上穿/下穿 BBI 则产生买入/卖出信号。

    指标描述：BBI=(MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4
    """
    # ==策略参数
    ma_short = int(para[0])  # 短期均线。ma代表：moving_average
    ma_middle1 = int(para[1])  # 长期均线
    ma_middle2 = int(para[2])  # 中期均线
    ma_long = int(para[3])  # 长期均线

    # ==计算指标
    df['ma_short'] = df['资金曲线'].rolling(ma_short, min_periods=1).mean()
    df['ma_middle1'] = df['资金曲线'].rolling(ma_middle1, min_periods=1).mean()
    df['ma_middle2'] = df['资金曲线'].rolling(ma_middle2, min_periods=1).mean()
    df['ma_long'] = df['资金曲线'].rolling(ma_long, min_periods=1).mean()
    df['BBI'] = (df['ma_short'] + df['ma_middle1'] + df['ma_middle2'] + df['ma_long']) / 4

    # ==找出做多信号
    con1 = df['资金曲线'].shift(1) <= df['BBI']  # 昨日资金曲线小于或等于BBI
    con2 = df['资金曲线'] > df['BBI']  # 今日资金曲线大于BBI
    df.loc[con1 & con2, 'signal'] = 1  # 产生做多信号1

    # ==找出做多平仓信号
    con1 = df['资金曲线'].shift(1) >= df['BBI']  # 昨日资金曲线大于或等于BBI
    con2 = df['资金曲线'] < df['BBI']  # 今日资金曲线小于BBI
    df.loc[con1 & con2, 'signal'] = 0  # 产生平仓信号0

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['ma_short', 'ma_middle1', 'ma_middle2', 'ma_long', 'BBI'], axis=1, inplace=True)

    return df, latest_signal


def PMO_signal(df, para=(2, 3, 5)):
    """
    https://bbs.quantclass.cn/thread/30069
    PMO策略。只能做多。
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param df:
    :param para: N1, N2, N3 是针对每日涨幅的连续三次平滑移动的参数
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    N1 = int(para[0])
    N2 = int(para[1])
    N3 = int(para[2])

    # ===计算PMO 和 PMO_SIGNAL。用资金曲线复权是否更加合理
    df['ROC'] = (df['资金曲线']-df['资金曲线'].shift(1))/df['资金曲线'].shift(1)*100
    df['ROC_MA'] = df['ROC'].rolling(N1, min_periods=1).mean()
    df['ROC_MA10'] = df['ROC_MA']*10
    df['PMO'] = df['ROC_MA10'].rolling(N2, min_periods=1).mean()
    df['PMO_SIGNAL'] = df['PMO'].rolling(N3, min_periods=1).mean()

    # ===找出做多信号
    condition1 = df['PMO'] > df['PMO_SIGNAL']  # 短期均线 > 长期均线
    condition2 = df['PMO'].shift(1) <= df['PMO_SIGNAL'].shift(1)  # 上一周期的短期均线 <= 长期均线
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['PMO'] < df['PMO_SIGNAL']  # 短期均线 < 长期均线
    condition2 = df['PMO'].shift(1) >= df['PMO_SIGNAL'].shift(1)   # 上一周期的短期均线 >= 长期均线
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号的那根K线的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['ROC', 'ROC_MA', 'ROC_MA10', 'PMO', 'PMO_SIGNAL'], axis=1, inplace=True)

    return df, latest_signal


def PO_signal(df, para=(2, 3)):
    """
    https://bbs.quantclass.cn/thread/31237
    :param df:
    :param para:
    :return:
    """

    # ===策略参数
    po_short = para[0]
    po_long = para[1]

    # ===计算po指标。所有的指标，都要使用复权价格进行计算。
    df['ema_short'] = df['资金曲线'].ewm(span=po_short, adjust=False).mean()
    df['ema_long'] = df['资金曲线'].ewm(span=po_long, adjust=False).mean()
    df['po'] = (df['ema_short'] - df['ema_long']) / df['ema_long'] * 100

    # ===找出做多信号
    condition1 = df['po'] > 0  # 本周期po>0
    condition2 = df['po'].shift(1) <= 0  # 上一周期的po<=0
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['po'] < 0  # 本周期po<0
    condition2 = df['po'].shift(1) >= 0  # 上一周期的po>=0
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['ema_short', 'ema_long', 'po'], axis=1, inplace=True)

    return df, latest_signal


# WMA 策略
def WMA_signal(df, para=6):
    """
    https://bbs.quantclass.cn/thread/30803
    WMA移动平均线计算择时信号,
    研报中的REF为计算一天前资金曲线
    :param df:
    :param para:
    :return:
    """
    n = para
    df['WMA'] = df['资金曲线'].rolling(n).apply(lambda x: x[::-1].cumsum().sum() * 2 / n / (n + 1))

    # print (df['资金曲线'])
    # ===找出做多信号
    condition1 = df['资金曲线'] > df['WMA']  # 资金曲线 > WMA
    condition2 = df['资金曲线'].shift(1) <= df['WMA'].shift(1)  # 上一周期的短期均线 <= 长期均线
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['资金曲线'] < df['WMA']  # 资金曲线 < WMA
    condition2 = df['资金曲线'].shift(1) >= df['WMA'].shift(1)  # 上一周期的短期均线 >= 长期均线
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['WMA'], axis=1, inplace=True)

    return df, latest_signal


def TMA_signal(df, para=(2, 4)):
    """
    https://bbs.quantclass.cn/thread/29734
    TMA策略。只能做多。
    N1=20, N2=20
    CLOSE_MA=MA(CLOSE,N1)
    TMA=MA(CLOSE_MA,N2)
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param df:
    :param para: N1, N2
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    N1 = int(para[0])  # 短期均线。ma代表：moving_average
    N2 = int(para[1])  # 长期均线

    # ===计算均线。所有的指标，都要使用复权价格进行计算。
    df['CLOSE_MA'] = df['资金曲线'].rolling(N1, min_periods=1).mean()
    df['TMA'] = df['CLOSE_MA'].rolling(N2, min_periods=1).mean()

    # ===找出做多信号
    condition1 = df['资金曲线'].shift(1) <= df['TMA'].shift(1)  # 前一日资金曲线小于TMA
    condition2 = df['资金曲线'] > df['TMA']  # 当日资金曲线大于TMA
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['资金曲线'].shift(1) >= df['TMA'].shift(1)  # 前一日资金曲线大于TMA
    condition2 = df['资金曲线'] < df['TMA']  # 当日资金曲线小于TMA
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['TMA', 'CLOSE_MA'], axis=1, inplace=True)

    return df, latest_signal


def MACD_signal(df, para=(2, 3, 5)):
    """
    https://bbs.quantclass.cn/thread/29790
    指标名称 MACD
    MACD=EMA(CLOSE,N1)-EMA(CLOSE,N2)
    MACD_SIGNAL=EMA(MACD,N3)
    MACD_HISTOGRAM=MACD-MACD_SIGNAL
    MACD 指标衡量快速均线与慢速均线的差值。由于慢速均线反映的是
    之前较长时间的价格的走向，而快速均线反映的是较短时间的价格的
    走向，所以在上涨趋势中快速均线会比慢速均线涨的快，而在下跌趋
    势中快速均线会比慢速均线跌得快。所以 MACD 上穿/下穿 0 可以作
    为一种构造交易信号的方式。另外一种构造交易信号的方式是求
    MACD 与其移动平均（信号线）的差值得到 MACD 柱，利用 MACD
    柱上穿/下穿 0（即 MACD 上穿/下穿其信号线）来构造交易信号。这
    种方式在其他指标的使用中也可以借鉴。"""
    # ===策略参数
    N1 = int(para[0])
    N2 = int(para[1])
    N3 = int(para[2])

    # ===计算MACD。所有的指标，都要使用复权价格进行计算。
    df['EMA_1'] = df['资金曲线'].ewm(span=N1, adjust=False).mean()  #EMA(CLOSE,N1)
    df['EMA_2'] = df['资金曲线'].ewm(span=N2, adjust=False).mean()  #EMA(CLOSE,N1)
    df['MACD'] = df['EMA_1'] - df['EMA_2']
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=N3, adjust=False).mean()  #EMA(MACD,N3)
    df['MACD_HISTOGRAM'] = df['MACD'] - df['MACD_SIGNAL']  # MACD_HISTOGRAM=MACD-MACD_SIGNAL

    # ===找出做多信号
    condition1 = df['MACD_HISTOGRAM'] > 0  # 利用 MACD柱上穿 0（即 MACD 上穿其信号线）来构造交易信号，买入
    condition2 = df['MACD_HISTOGRAM'].shift(1) < 0
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['MACD_HISTOGRAM'] < 0  # 利用 MACD柱下穿 0（即 MACD 下穿其信号线）来构造交易信号，卖出
    condition2 = df['MACD_HISTOGRAM'].shift(1) > 0
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['EMA_1', 'EMA_2'], axis=1, inplace=True)

    return df, latest_signal


# KDJ策略
def KDJ_signal(df, para=(2, 3, 4)):
    """
    https://bbs.quantclass.cn/thread/30081
    简单的移动平均线策略。只能做多。
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param df:
    :param para:
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    N1 = int(para[0])
    N2 = int(para[1])
    N3 = int(para[2])

    # ===计算过去 N 天指标 X 的最大值，最小值
    df['MAX_rolling'] = df['资金曲线'].rolling(N1).max()
    df['MIN_rolling'] = df['资金曲线'].rolling(N1).min()

    # ===计算Stochastics
    df['Stochastics'] = (df['资金曲线'] - df['MIN_rolling'])/(df['MAX_rolling'] - df['MIN_rolling']) * 100

    # ===计算K,D
    df['K'] = df['Stochastics'].ewm(span=(3 - 1), adjust=False).mean()
    df['D'] = df['K'].ewm(span=(3 - 1), adjust=False).mean()

    # ===找出做平仓信号
    condition1 = df['K'] > df['D']  # K > D
    condition2 = df['D'] < N2
    df.loc[condition1 & condition2, 'signal'] = 1

    # ===找出做多信号
    condition1 = df['K'] < df['D']  # K < D
    condition2 = df['D'] > N3
    df.loc[condition1 & condition2, 'signal'] = 0

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['K', 'D'], axis=1, inplace=True)

    return df, latest_signal


def ARRON_signal(df, para=(2, 5)):
    """
    https://bbs.quantclass.cn/thread/30526
    研究当前股价与历史最/最低价之间的关系来发出交易信号。
    Arron指数离 0 越远，趋势越强。
    当Arron指标大于买入阈值时买入，小于卖出阈值时卖出。
    :param df:
    :param para: N, buy_Con, sale_Con
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    N = para[0]  # 周期长度
    buy_Con = para[1]  # 买入信号发出的阈值（正数）
    sell_Con = -para[1]  # 卖出信号发出的阈值（负数）

    # ==找出最大值行号
    def High_len(df):  # 计算最大值天数差值
        return len(df) - np.argmax(df)-1

    def High_index(df):  # 找出最大值行号
        # print(np.argmax(df))
        return df[np.argmax(df)]

    def Low_len(df):  # 计算最小值天数差值
        # print(df)
        return len(df) - np.argmin(df)-1

    def Low_index(df):  # 找出最小值行号
        # print(df)
        return df[np.argmin(df)]

    # ==计算ArronOs指标
    # 区间最高价行号
    df['High_index'] = df['资金曲线'].rolling(N, min_periods=1).apply(High_index, raw=True)
    # 区间最高价距离当日天数
    df['High_len'] = df['资金曲线'].rolling(N, min_periods=1).apply(High_len, raw=True)
    # 区间最低价行号
    df['Low_index'] = df['资金曲线'].rolling(N, min_periods=1).apply(Low_index, raw=True)
    # 区间最低价距离当前天数
    df['Low_len'] = df['资金曲线'].rolling(N, min_periods=1).apply(Low_len, raw=True)

    df['Arron_Up'] = (N - df['High_len']) / N * 100
    df['Arron_Down'] = (N - df['Low_len']) / N * 100
    df['Arron_Os'] = df['Arron_Up'] - df['Arron_Down']
    # 找出买入信号
    condition01 = df['Arron_Os'] >= buy_Con
    condition02 = df['Arron_Os'].shift(1) < buy_Con
    df.loc[condition01 & condition02, 'signal'] = 1
    # 找出卖出信号
    condition01 = df['Arron_Os'] <= sell_Con
    condition02 = df['Arron_Os'].shift(1) > sell_Con
    df.loc[condition01 & condition02, 'signal'] = 0

    df['signal'].fillna(method='ffill', inplace=True)
    df['signal'].fillna(value=0, inplace=True)  # 将初始行数的signal补全为0

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    df.drop(['High_index', 'High_len', 'Low_len', 'Low_index', 'Arron_Up', 'Arron_Down', 'Arron_Os'], axis=1,
            inplace=True)

    return df, latest_signal
