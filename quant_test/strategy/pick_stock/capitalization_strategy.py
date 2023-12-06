import pandas as pd

from strategy.strategy_utils import *


def small_cap_strategy(pick_from_df, select_stock_num):
    """
    小市值策略
    :param select_stock_num: 选股数
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100003  # 代表选股策略中的第三个

    pick_from_df = rule_out_stocks_global(pick_from_df)

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    pick_from_df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, pick_from_df


def monthly_cap_strategy(pick_from_df, select_stock_num):
    """
    根据月份应用大市值或小市值选股策略 by GPT4.0
    :param pick_from_df: 用于选股的数据
    :param select_stock_num: 选股数
    :return: 经过筛选的股票数据
    """
    session_id = 100020

    # 提取月份信息
    pick_from_df['月份'] = pick_from_df['交易日期'].dt.month

    # 筛选1月和4月的数据
    large_cap_df = pick_from_df[pick_from_df['月份'].isin([1, 4])]

    # 筛选除1月和4月外的其他月份的数据
    small_cap_df = pick_from_df[~pick_from_df['月份'].isin([1, 4])]

    # 应用大市值策略
    _, large_cap_selected = large_cap_strategy(large_cap_df, select_stock_num)

    # 应用小市值策略
    _, small_cap_selected = small_cap_strategy(small_cap_df, select_stock_num)

    # 合并结果并按照交易日期排序
    combined_df = pd.concat([large_cap_selected, small_cap_selected]).sort_values(by='交易日期')

    return session_id, combined_df


def relative_small_cap_strategy(pick_from_df, select_stock_num):
    """
    相对小市值策略，排名不选最低，而是从一个特定位置开始
    :param select_stock_num: 选股数
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100016

    pick_from_df = rule_out_stocks_global(pick_from_df)

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

    pick_from_df = rule_out_stocks_global(pick_from_df)
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
    在3的基础上，依据近期数据进行调整
    :param pick_from_df: 选股数据
    :param select_stock_num: 选股数
    :return:
    """
    session_id = 100015

    pick_from_df = rule_out_stocks_global(pick_from_df)
    df = pick_from_df

    # 筛选：过滤掉市值太大的，保留小市值这一范围
    df = df[df['总市值 （万元）'] < 300000]

    # 计算所需因子排名
    df['量价相关性_10排名百分比'] = df.groupby('交易日期')['量价相关性_10'].rank(ascending=True, pct=True, method='min')
    df['成交额std_10排名百分比'] = df.groupby('交易日期')['成交额std_10'].rank(ascending=True, pct=True, method='min')

    # 筛选：过滤掉量价相关性_20最大的30%
    df = df[df['量价相关性_10排名百分比'] < 0.7]
    # 筛选：过滤掉成交额std_20最大的25%
    df = df[df['成交额std_10排名百分比'] < 0.75]

    # 排序：
    df['非流动性排名'] = df.groupby('交易日期')['非流动性_10'].rank(ascending=False, pct=False, method='min')
    df['市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='min')
    df['量价排名'] = df.groupby('交易日期')['量价相关性_10'].rank(ascending=True, pct=False, method='min')

    # 计算复合因子
    df['复合因子'] = df['量价排名'] + df['非流动性排名'] + df['市值排名'] * 2
    # 对因子进行排名
    df['排名'] = df.groupby('交易日期')['复合因子'].rank()
    # 选取排名靠前的股票
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def small_cap_bin_optimized3(pick_from_df, select_stock_num):
    """
    小市值策略+分箱优化3
    使用单调性好的因子进行筛选
    完全使用全量数据
    :param pick_from_df: 选股数据
    :param select_stock_num: 选股数
    :return:
    """
    session_id = 100015

    pick_from_df = rule_out_stocks_global(pick_from_df)
    df = pick_from_df

    # 筛选：过滤掉市值太大的，保留小市值这一范围
    df = df[df['总市值 （万元）'] < 300000]

    # 计算所需因子排名
    df['量价相关性_20排名百分比'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=True, method='min')
    df['成交额std_20排名百分比'] = df.groupby('交易日期')['成交额std_20'].rank(ascending=True, pct=True, method='min')

    # 筛选：过滤掉量价相关性_20最大的35%
    df = df[df['量价相关性_20排名百分比'] < 0.65]
    # 筛选：过滤掉成交额std_20最大的25%
    df = df[df['成交额std_20排名百分比'] < 0.75]

    # 排序：
    df['非流动性排名'] = df.groupby('交易日期')['非流动性_5'].rank(ascending=False, pct=False, method='min')
    df['振幅排名'] = df.groupby('交易日期')['振幅_20'].rank(ascending=False, pct=False, method='min')
    df['市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='min')
    df['量价排名'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=False, method='min')

    # 计算复合因子
    df['复合因子'] = df['量价排名'] + df['非流动性排名'] + df['振幅排名'] + df['市值排名'] * 2
    # 对因子进行排名
    df['排名'] = df.groupby('交易日期')['复合因子'].rank()
    # 选取排名靠前的股票
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def small_cap_bin_optimized4(pick_from_df, select_stock_num):
    """
    小市值策略+分箱优化4
    基于3的优化：使用单调性好的因子：用短时间数据过滤，用长时间数据排名
    :param pick_from_df: 选股数据
    :param select_stock_num: 选股数
    :return:
    """
    session_id = 100015

    pick_from_df = rule_out_stocks_global(pick_from_df)
    df = pick_from_df

    # 筛选：过滤掉市值太大的，保留小市值这一范围
    df = df[df['总市值 （万元）'] < 300000]

    # 计算所需因子排名
    df['量价相关性_10排名百分比'] = df.groupby('交易日期')['量价相关性_10'].rank(ascending=True, pct=True, method='min')
    df['成交额std_10排名百分比'] = df.groupby('交易日期')['成交额std_10'].rank(ascending=True, pct=True, method='min')
    df['alpha95排名百分比'] = df.groupby('交易日期')['alpha95'].rank(ascending=True, pct=True, method='min')

    # 筛选：过滤掉量价相关性_10最大的30%
    df = df[df['量价相关性_10排名百分比'] < 0.7]
    # 筛选：过滤掉成交额std_10最大的25%
    df = df[df['成交额std_10排名百分比'] < 0.75]
    # 筛选：过滤掉alpha95最大的20%
    df = df[df['alpha95排名百分比'] < 0.8]

    # 排序：
    df['非流动性排名'] = df.groupby('交易日期')['非流动性_5'].rank(ascending=False, pct=False, method='min')
    df['市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='min')
    df['振幅排名'] = df.groupby('交易日期')['振幅_20'].rank(ascending=False, pct=False, method='min')
    df['量价排名'] = df.groupby('交易日期')['量价相关性_20'].rank(ascending=True, pct=False, method='min')

    # 计算复合因子
    df['复合因子'] = df['量价排名'] + df['非流动性排名'] + df['振幅排名'] + df['市值排名'] * 2
    # 对因子进行排名
    df['排名'] = df.groupby('交易日期')['复合因子'].rank()
    # 选取排名靠前的股票
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def small_cap_bin_optimized5(pick_from_df, select_stock_num):
    """
    小市值策略+分箱优化5
    使用单调性好的因子排序
    并且排除分箱效果太差的
    根据最近数据优化
    :param pick_from_df: 选股数据
    :param select_stock_num: 选股数
    :return:
    """
    session_id = 100015

    pick_from_df = rule_out_stocks_global(pick_from_df)
    df = pick_from_df

    # 筛选：过滤掉市值太大的，保留小市值这一范围
    df = df[df['总市值 （万元）'] < 300000]

    # 计算所需因子排名
    df['alpha95排名百分比'] = df.groupby('交易日期')['alpha95'].rank(ascending=True, pct=True, method='min')
    df['前日成交额排名百分比'] = df.groupby('交易日期')['前日成交额'].rank(ascending=True, pct=True, method='min')
    df['量价相关性_10排名百分比'] = df.groupby('交易日期')['量价相关性_10'].rank(ascending=True, pct=True, method='min')
    df['成交额std_10排名百分比'] = df.groupby('交易日期')['成交额std_10'].rank(ascending=True, pct=True, method='min')

    # 筛选：过滤掉alpha95最大的25%
    df = df[df['alpha95排名百分比'] < 0.75]
    # 筛选：过滤掉前日成交额最大的20%
    df = df[df['前日成交额排名百分比'] < 0.8]
    # 筛选：过滤掉量价相关性_10最大的25%
    df = df[df['量价相关性_10排名百分比'] < 0.75]
    # 筛选：过滤掉成交额std_10最大的15%
    df = df[df['成交额std_10排名百分比'] < 0.85]

    # 排序：
    df['非流动性排名'] = df.groupby('交易日期')['非流动性_10'].rank(ascending=False, pct=False, method='min')
    df['市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='min')
    df['量价排名'] = df.groupby('交易日期')['量价相关性_10'].rank(ascending=True, pct=False, method='min')

    # 计算复合因子
    df['复合因子'] = df['量价排名'] + df['非流动性排名'] + df['市值排名']
    # 对因子进行排名
    df['排名'] = df.groupby('交易日期')['复合因子'].rank()

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

    pick_from_df = rule_out_stocks_global(pick_from_df)

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
    :param select_stock_num: 选股数
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100007

    pick_from_df = rule_out_stocks_global(pick_from_df)

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

    pick_from_df = rule_out_stocks_global(pick_from_df)

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

    pick_from_df = rule_out_stocks_global(pick_from_df)

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

    pick_from_df = rule_out_stocks_global(pick_from_df)

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

    pick_from_df = rule_out_stocks_global(pick_from_df)

    pick_from_df = pick_from_df[pick_from_df['收盘价'] >= 2]
    # 低价股 + 小市值选股：混合排名
    pick_from_df['价格排名'] = pick_from_df.groupby('交易日期')['收盘价'].rank(ascending=True, pct=False, method='first')
    pick_from_df['市值排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, pct=False, method='first')

    # 选股
    pick_from_df['因子'] = pick_from_df['价格排名'] + pick_from_df['市值排名']
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['因子'].rank()
    pick_from_df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, pick_from_df
