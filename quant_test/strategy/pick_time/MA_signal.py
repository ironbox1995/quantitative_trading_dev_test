# MA策略
def MA_signal(select_stock, para=(1, 3)):
    """
    简单的移动平均线策略。只能做多。
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param select_stock:
    :param para: ma_short, ma_long
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

    # ===策略参数
    ma_short = para[0]  # 短期均线。ma代表：moving_average
    ma_long = para[1]  # 长期均线

    # ===计算均线。所有的指标，都要使用复权价格进行计算。
    select_stock['ma_short'] = select_stock['资金曲线'].rolling(ma_short, min_periods=1).mean()
    select_stock['ma_long'] = select_stock['资金曲线'].rolling(ma_long, min_periods=1).mean()

    # ===找出做多信号
    condition1 = select_stock['ma_short'] > select_stock['ma_long']  # 短期均线 > 长期均线
    condition2 = select_stock['ma_short'].shift(1) <= select_stock['ma_long'].shift(1)  # 上一周期的短期均线 <= 长期均线
    select_stock.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

    # ===找出平仓信号
    condition1 = select_stock['ma_short'] < select_stock['ma_long']  # 短期均线 < 长期均线
    condition2 = select_stock['ma_short'].shift(1) >= select_stock['ma_long'].shift(1)  # 上一周期的短期均线 >= 长期均线
    select_stock.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    select_stock['signal'].fillna(method='ffill', inplace=True)
    latest_signal = select_stock.tail(1)['signal'].iloc[0]
    select_stock['signal'] = select_stock['signal'].shift(1)  # 产生的信号下个周期才能用
    select_stock['signal'].fillna(value=1, inplace=True)  # 最前面正常买入即可
    # select_stock['signal'].fillna(method='ffill', inplace=True)


    # ===删除无关中间变量
    select_stock.drop(['ma_short', 'ma_long'], axis=1, inplace=True)

    return select_stock, latest_signal
