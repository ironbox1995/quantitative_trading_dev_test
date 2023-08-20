import pandas as pd
import numpy as np


# =========MICD 异同离差动力指数========
# MICD策略
def MICD_signal(select_stock, para=(4, 2)):
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
def SROC_signal(df, para=(3, 4, 5)):
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
    condition2 = df['资金曲线'].shift(1) <= df['UPPER'].shift(1)  # 上一周期的收盘价 <= 上轨线
    df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出做多平仓信号
    condition1 = df['资金曲线'] < df['LOWER']  # 资金曲线 < 下轨线
    condition2 = df['资金曲线'].shift(1) >= df['LOWER'].shift(1)  # 上一周期的收盘价 >= 下轨线
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    df['signal'].fillna(method='ffill', inplace=True)
    latest_signal = df.tail(1)['signal'].iloc[0]
    df['signal'] = df['signal'].shift(1)  # 产生的信号下个周期才能用
    df['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可

    # ===删除无关中间变量
    df.drop(['MAC', 'UPPER', 'LOWER'], axis=1, inplace=True)

    return df, latest_signal


# MTM策略
def MTM_signal(df, para=2):
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
    DPO=CLOSE-REF(MA(CLOSE,N),N/2+1),收盘价 - N/2+1天前的 N周期均线
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
