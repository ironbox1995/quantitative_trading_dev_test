import pandas as pd


def small_cap_strategy(pick_from_df, select_stock_num):
    """
    小市值策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100003  # 代表选股策略中的第三个

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df


def small_cap_strategy_pv_opt_1(pick_from_df, select_stock_num):
    """
    小市值策略+量价优化1
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100003  # 代表选股策略中的第三个

    # 计算总市值排名
    pick_from_df['总市值排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, method='min')
    # 计算alpha95排名  衡量低波动
    pick_from_df['alpha95排名'] = pick_from_df.groupby('交易日期')['alpha95'].rank(ascending=True, method='min')
    # 计算Ret20排名  衡量超跌
    pick_from_df['20日涨跌幅排名'] = pick_from_df.groupby('交易日期')['20日涨跌幅'].rank(ascending=True, method='min')
    # 计算复合因子
    pick_from_df['复合因子'] = pick_from_df['alpha95排名'] + pick_from_df['总市值排名'] + pick_from_df['20日涨跌幅排名']
    # 对因子进行排名
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['复合因子'].rank()
    # 选取排名靠前的股票
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df


def large_cap_strategy(pick_from_df, select_stock_num):
    """
    大市值策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100007

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df


def low_price_strategy(pick_from_df, select_stock_num=50):
    """
    低价选股策略
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100013

    # 低价股选股：价格最低的50只(默认值为50)，且收盘价>=2
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=False, method='first')
    df = pick_from_df[(pick_from_df['价格排名'] <= select_stock_num) & (pick_from_df['收盘价'] >= 2)]

    return session_id, df


def low_price_pct_strategy(pick_from_df, select_stock_num):
    """
    低价选股策略_百分比
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100014

    # print("本策略不需要select_stock_num：{}".format(select_stock_num))
    # 低价股选股：价格最低的前20%只股票，且收盘价>=2
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=True, method='first')
    df = pick_from_df[(pick_from_df['价格排名'] <= 0.2) & (pick_from_df['收盘价'] >= 2)]

    return session_id, df


def low_price_small_cap_strategy(pick_from_df, select_stock_num=200):
    """
    垃圾股策略：低价股 + 小市值选股
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100015

    # 低价股 + 小市值选股：价格和市值同时满足：最小的前200只股票，且收盘价>=2
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=False, method='first')
    pick_from_df['市值排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='first')
    df = pick_from_df[(pick_from_df['价格排名'] <= select_stock_num) & (pick_from_df['市值排名'] <= select_stock_num)
                      & (pick_from_df['收盘价'] >= 2)]

    return session_id, df
