from Config.global_config import *


def small_cap_strategy(pick_from_df, select_stock_num):
    """
    小市值策略
    :param select_stock_num: 选股数
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100003  # 代表选股策略中的第三个

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    pick_from_df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, pick_from_df


def relative_small_cap_strategy(pick_from_df, select_stock_num):
    """
    相对小市值策略，排名不选最低，而是从一个特定位置开始
    :param select_stock_num: 选股数
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100016

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    start_point = 1  # 取1时，和原来的小市值策略相同

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    pick_from_df = pick_from_df[start_point <= pick_from_df['排名'] < start_point + select_stock_num]

    return session_id, pick_from_df


def small_cap_bin_optimized1(pick_from_df, select_stock_num):
    """
    小市值策略+分箱优化1
    :param pick_from_df: 选股数据
    :param select_stock_num: 选股数
    :return:
    """
    session_id = 100015

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行
    df = pick_from_df

    # 筛选：过滤掉市值太大的，保留小市值这一范围
    df = df[df['总市值 （万元）'] < 300000]

    # 计算所需因子排名
    df['量比排名百分比'] = df.groupby('交易日期')['量比'].rank(ascending=True, pct=True, method='min')
    # df['ROC_20排名百分比'] = df.groupby('交易日期')['ROC_20'].rank(ascending=True, pct=True, method='min')
    df['成交额std_20排名百分比'] = df.groupby('交易日期')['成交额std_20'].rank(ascending=True, pct=True, method='min')
    df['量价相关性_20排名百分比'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=True, method='min')

    # 筛选：过滤掉量比最大的25%
    df = df[df['量比排名百分比'] < 0.75]
    # # 筛选：保留ROC_20最小的10%
    # df = df[df['ROC_20排名百分比'] < 0.10]
    # 筛选：过滤掉成交额std_20最大的25%
    df = df[df['成交额std_20排名百分比'] < 0.75]
    # 筛选：过滤掉量价相关性_20最大的25%
    df = df[df['量价相关性_20排名百分比'] < 0.75]

    # 排序：
    df['振幅排名'] = df.groupby('交易日期')['振幅_20'].rank(ascending=False, pct=False, method='min')
    df['市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='min')
    df['量价排名'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=False, method='min')

    # 计算复合因子
    df['复合因子'] = df['市值排名'] + df['量价排名'] + df['振幅排名']
    # 对因子进行排名
    df['排名'] = df.groupby('交易日期')['复合因子'].rank()
    # 选取排名靠前的股票
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def small_cap_bin_optimized2(pick_from_df, select_stock_num):
    """
    小市值策略+分箱优化2
    :param pick_from_df: 选股数据
    :param select_stock_num: 选股数
    :return:
    """
    session_id = 100015

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行
    df = pick_from_df

    # 筛选：过滤掉市值太大的，保留小市值这一范围
    df = df[df['总市值 （万元）'] < 300000]

    # 计算所需因子排名
    # df['量比排名百分比'] = df.groupby('交易日期')['量比'].rank(ascending=True, pct=True, method='min')
    df['ROC_20排名百分比'] = df.groupby('交易日期')['ROC_20'].rank(ascending=True, pct=True, method='min')
    # df['成交额std_20排名百分比'] = df.groupby('交易日期')['成交额std_20'].rank(ascending=True, pct=True, method='min')
    # df['量价相关性_20排名百分比'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=True, method='min')

    # 筛选：过滤掉量比最大的25%
    # df = df[df['量比排名百分比'] < 0.75]
    # # 筛选：保留ROC_20最小的10%
    df = df[df['ROC_20排名百分比'] < 0.10]
    # 筛选：过滤掉成交额std_20最大的25%
    # df = df[df['成交额std_20排名百分比'] < 0.75]
    # 筛选：过滤掉量价相关性_20最大的25%
    # df = df[df['量价相关性_20排名百分比'] < 0.75]

    # 排序：
    # df['振幅排名'] = df.groupby('交易日期')['振幅_20'].rank(ascending=False, pct=False, method='min')
    df['市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='min')
    # df['量价排名'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=False, method='min')

    # 计算复合因子
    # df['复合因子'] = df['市值排名'] + df['量价排名'] + df['振幅排名']
    # 对因子进行排名
    df['排名'] = df.groupby('交易日期')['市值排名'].rank()
    # 选取排名靠前的股票
    df = df[df['排名'] <= select_stock_num]

    return session_id, df



def small_cap_strategy_pv_opt_1(pick_from_df, select_stock_num):
    """
    小市值策略+量价优化1
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100003  # 代表选股策略中的第三个

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # print("本策略不需要select_stock_num：{}".format(select_stock_num))
    # 低价股选股：价格最低的前20%只股票，且收盘价>=2
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=True, method='first')
    df = pick_from_df[(pick_from_df['价格排名'] <= 0.2) & (pick_from_df['收盘价'] >= 2)]

    return session_id, df


def junk_stock_strategy(pick_from_df, select_stock_num=200):
    """
    垃圾股策略：低价股 + 小市值选股
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """

    session_id = 100015

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # 低价股 + 小市值选股：价格和市值同时满足：最小的前200只股票，且收盘价>=2
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=False, method='first')
    pick_from_df['市值排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='first')
    df = pick_from_df[(pick_from_df['价格排名'] <= select_stock_num) & (pick_from_df['市值排名'] <= select_stock_num)
                      & (pick_from_df['收盘价'] >= 2)]

    return session_id, df


def low_price_small_cap_strategy(pick_from_df, select_stock_num):
    """
    低价小市值策略
    低价因子+小市值因子
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100015

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    pick_from_df = pick_from_df[pick_from_df['收盘价'] >= 2]
    # 低价股 + 小市值选股：混合排名
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=False, method='first')
    pick_from_df['市值排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='first')

    # 选股
    pick_from_df['因子'] = pick_from_df['价格排名'] + pick_from_df['市值排名']
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['因子'].rank()
    pick_from_df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, pick_from_df
