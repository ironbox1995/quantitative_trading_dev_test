def price_volume_strategy(df, select_stock_num):
    factor = '量价相关性'  # 量价相关性
    ascending = True  # True，从小到大    False，从大到小
    df.dropna(subset=[factor], inplace=True)
    df['排名'] = df.groupby('交易日期')[factor].rank(ascending=ascending, method='first')
    df['排名_百分比'] = df.groupby('交易日期')[factor].rank(ascending=ascending, pct=True, method='first')

    # ===选股
    if select_stock_num:
        if select_stock_num >= 1:
            df = df[df['排名'] <= select_stock_num]
        else:
            df = df[df['排名_百分比'] <= select_stock_num]

    session_id = 100011

    return session_id, df


def volume_ratio_strategy(pick_from_df, select_stock_num):
    """
    量比选股策略 （第一个自己的策略，这个策略应该只能按周轮动甚至按日轮动）
    https://zhuanlan.zhihu.com/p/62167733
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100017

    # 条件：量比大于2.5 而 小于 5
    condition = (pick_from_df['量比'] > 2.5) & (pick_from_df['量比'] <= 5)
    # 条件：换手率小于3%
    condition &= (pick_from_df['当日换手率'] < 3)
    # 条件：流通股本在3亿以下
    condition &= (pick_from_df['流通市值（万元）']/pick_from_df['收盘价'] < 30000)
    # 条件：中小板
    condition &= (pick_from_df['市场类型'] == '中小板')

    # 根据条件进行选股
    pick_from_df = pick_from_df[condition]

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['量比'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df


def wr_bias_strategy(pick_from_df, select_stock_num):
    """
    不知道这个策略该叫什么，姑且称为：香农短线量价策略
    https://bbs.quantclass.cn/thread/12943
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100018

    df = pick_from_df
    # 筛选
    df = df[df['WR_5'] >= 15]
    df = df[(df['bias_5'] >= -0.05)]

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['总市值_排名'] = df.groupby('交易日期')['总市值'].rank()
    df['bias_10_排名'] = df.groupby('交易日期')['bias_10'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['总市值_排名'] + df['bias_10_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def volume_turnover_rate_strategy(pick_from_df, select_stock_num):
    """
    挖掘放量待涨个股策略
    https://bbs.quantclass.cn/thread/13266
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100019

    df = pick_from_df

    # 筛选
    df = df[df['换手率'] > 0.035]
    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    # df['换手率mean_10_排名'] = df.groupby('交易日期')['换手率mean_10'].rank()  # 可以把上一行换成这个试试
    df['成交额_排名'] = df.groupby('交易日期')['成交额'].rank()
    df['涨跌幅_排名'] = df.groupby('交易日期')['涨跌幅'].rank()

    # 选股

    df['因子'] = df['成交额std_10_排名'] + 0.72 * df['成交额_排名'] + 0.50 * df['涨跌幅_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df
