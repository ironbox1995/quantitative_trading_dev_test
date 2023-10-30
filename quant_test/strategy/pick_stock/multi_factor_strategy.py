from utils_global.global_config import *


def factor_iterated_strategy1(pick_from_df, select_stock_num):
    """
    因子遍历增强策略1
    https://bbs.quantclass.cn/thread/2513
    原文选择了10支股票，这里我仍然选三支
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100030

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    df = pick_from_df
    df = df[df['市盈率'] > 0]  # 此处有改动

    # 排序
    factor_list = [
        '成交额std_10',
        '成交额_10',
        '成交额std_20',
        '成交额std_5',
        '总市值 （万元）',
    ]
    for factor in factor_list:
        df[factor + '_排名'] = df.groupby('交易日期')[factor].rank()

    df['因子'] = 0
    for factor in factor_list:
        df['因子'] += df[factor + '_排名']

    # 选股
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def low_draw_down_factors_strategy(pick_from_df, select_stock_num):
    """
    低回撤单因子组合策略
    https://bbs.quantclass.cn/thread/2440
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100031

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    df = pick_from_df

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['总市值_排名'] = df.groupby('交易日期')['总市值 （万元）'].rank()
    df['成交额std_20_排名'] = df.groupby('交易日期')['成交额std_20'].rank()
    df['bias_5_排名'] = df.groupby('交易日期')['bias_5'].rank()
    # 选股
    df['因子'] = df['成交额std_10_排名'] + 0.999 * df['bias_5_排名'] + df['总市值_排名'] + df['成交额std_20_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def bias_and_circulating_value_strategy(pick_from_df, select_stock_num):
    """
    均线偏离与流通市值策略
    https://bbs.quantclass.cn/thread/2815
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100032

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    df = pick_from_df

    # 排序
    df['成交额std_10排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['成交额std_20排名'] = df.groupby('交易日期')['成交额std_20'].rank()
    df['bias_20_排名'] = df.groupby('交易日期')['bias_20'].rank()
    df['流通市值'] = df.groupby('交易日期')['流通市值（万元）'].rank()
    # 选股
    df['因子'] = df['成交额std_10排名'] + df['成交额std_20排名'] + df['bias_20_排名'] + df['流通市值']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def turnover_filter_strategy(pick_from_df, select_stock_num):
    """
    换手率筛选多因子排序策略
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100033

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    df = pick_from_df

    # 筛选
    df = df[(df['bias_5'] >= -0.05)]
    df = df[(df['成交额std_5'] <= df['成交额std_10']) & (df['成交额std_5'] <= df['成交额std_20'])]
    df = df[df['换手率'] < 0.258]
    df = df[df['换手率'] > 0.05]

    # 排序
    df['总市值_排名'] = df.groupby('交易日期')['总市值 （万元）'].rank()
    df['bias_5_排名'] = df.groupby('交易日期')['bias_5'].rank()
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['总市值_排名'] + df['bias_5_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df
