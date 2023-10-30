from utils_global.global_config import *


def price_volume_strategy(df, select_stock_num):
    factor = '量价相关性'  # 量价相关性
    ascending = True  # True，从小到大    False，从大到小
    df.dropna(subset=[factor], inplace=True)

    if not Second_Board_available:
        df = df[df['市场类型'] != '创业板']
    if not STAR_Market_available:
        df = df[df['市场类型'] != '科创板']
    if use_black_list:
        df = df[~df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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


def multi_factor_pv_strategy1(pick_from_df, select_stock_num):
    """
    多因子量价策略1
    https://zhuanlan.zhihu.com/p/62167733
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100017

    df = pick_from_df

    if not Second_Board_available:
        df = df[df['市场类型'] != '创业板']
    if not STAR_Market_available:
        df = df[df['市场类型'] != '科创板']
    if use_black_list:
        df = df[~df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # 筛选
    df['杠杆'] = df['流通市值（万元）'] / df['总市值 （万元）']
    df['杠杆_排名'] = df.groupby('交易日期')['杠杆'].rank(pct=True)
    df = df[df['杠杆_排名'] < 0.8]
    df['流通市值_排名'] = df.groupby('交易日期')['流通市值（万元）'].rank(pct=True)
    df = df[df['流通市值_排名'] < 0.6]
    df['量价相关性_20_排名'] = df.groupby('交易日期')['量价相关性_20'].rank(pct=True)
    df = df[df['量价相关性_20_排名'] < 0.8]
    df['量价相关性_10_排名'] = df.groupby('交易日期')['量价相关性_10'].rank(pct=True)
    df = df[df['量价相关性_10_排名'] < 0.8]
    df['均线_10_排名'] = df.groupby('交易日期')['均线_10'].rank(pct=True)
    df = df[df['均线_10_排名'] < 0.8]
    con1 = df['涨跌幅_5'] <= 0.15
    con3 = df['涨跌幅_20'] <= 0.3
    df = df[con1 & con3]

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['成交额std_5_排名'] = df.groupby('交易日期')['成交额std_5'].rank()
    df['成交额std_20_排名'] = df.groupby('交易日期')['成交额std_20'].rank()
    df['bias_10_排名'] = df.groupby('交易日期')['bias_10'].rank()
    df['流通市值_排名'] = df.groupby('交易日期')['流通市值（万元）'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['成交额std_5_排名'] + df['成交额std_20_排名'] + df['bias_10_排名'] + df['流通市值_排名']  #
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]
    return session_id, df


def multi_factor_pv_strategy2(pick_from_df, select_stock_num):
    """
    多因子量价策略2
    https://bbs.quantclass.cn/thread/2888
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100017

    df = pick_from_df

    if not Second_Board_available:
        df = df[df['市场类型'] != '创业板']
    if not STAR_Market_available:
        df = df[df['市场类型'] != '科创板']
    if use_black_list:
        df = df[~df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # 筛选
    df['量价相关性_20_排名'] = df.groupby('交易日期')['量价相关性_20'].rank(pct=True)
    df = df[df['量价相关性_20_排名'] < 0.8]

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['成交额std_5_排名'] = df.groupby('交易日期')['成交额std_5'].rank()
    df['成交额std_20_排名'] = df.groupby('交易日期')['成交额std_20'].rank()
    df['bias_10_排名'] = df.groupby('交易日期')['bias_10'].rank()
    df['流通市值_排名'] = df.groupby('交易日期')['流通市值（万元）'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['成交额std_5_排名'] + df['成交额std_20_排名'] + df['bias_10_排名'] + df['流通市值_排名']  #
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def non_high_price_strategy(pick_from_df, select_stock_num):
    """
    非高价股选股策略
    https://bbs.quantclass.cn/thread/2721
    :return:
    """
    session_id = 100020

    df = pick_from_df

    if not Second_Board_available:
        df = df[df['市场类型'] != '创业板']
    if not STAR_Market_available:
        df = df[df['市场类型'] != '科创板']
    if use_black_list:
        df = df[~df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    df = df[df['最高价'] < 45]
    df = df[df['最高价'] > 4]
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['总市值_排名'] = df.groupby('交易日期')['总市值 （万元）'].rank()
    df['成交额std_5_排名'] = df.groupby('交易日期')['成交额std_5'].rank()
    df['bias_5_排名'] = df.groupby('交易日期')['bias_5'].rank()
    df['因子'] = 0.45 * df['成交额std_10_排名'] + 0.4 * df['bias_5_排名'] + 0.45 * df['总市值_排名'] + 0.55 * df['成交额std_5_排名']  # 这尼玛参数看着就不靠谱
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

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

    if not Second_Board_available:
        df = df[df['市场类型'] != '创业板']
    if not STAR_Market_available:
        df = df[df['市场类型'] != '科创板']
    if use_black_list:
        df = df[~df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # 筛选
    df = df[df['WR_5'] >= 15]
    df = df[(df['bias_5'] >= -0.05)]

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['总市值_排名'] = df.groupby('交易日期')['总市值 （万元）'].rank()
    df['bias_10_排名'] = df.groupby('交易日期')['bias_10'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['总市值_排名'] + df['bias_10_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def volume_turnover_rate_strategy(pick_from_df, select_stock_num):
    """
    挖掘放量待涨小市值个股策略
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100019

    df = pick_from_df

    if not Second_Board_available:
        df = df[df['市场类型'] != '创业板']
    if not STAR_Market_available:
        df = df[df['市场类型'] != '科创板']
    if use_black_list:
        df = df[~df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # 排序
    df['总市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank()
    df['bias_5_排名'] = df.groupby('交易日期')['bias_5'].rank()

    # 选股

    df['因子'] = df['总市值排名'] + df['bias_5_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df
