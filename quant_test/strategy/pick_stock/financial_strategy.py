from strategy.strategy_utils import *


# def financial_report_strategy2(pick_from_df, select_stock_num):
#     """
#     https://bbs.quantclass.cn/thread/2685
#     财报严选财务策略2
#     :param buy_amount: 最大仓位
#     :param pick_from_df: 用于选股的数据
#     :return:
#     """
#     session_id = 100004
#
#     pick_from_df = rule_out_stocks_global(pick_from_df)
#
#     df = pick_from_df
#     # 筛选
#     df = df[df['营业收入_单季同比'] > 0]
#     df = df[df['净利润'] > 0]
#     df = df[df['经营活动产生的现金流量净额'] / df['净利润'] > 0.5]
#
#     # 排序
#     df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
#
#     # 选股
#     df['因子'] = df['成交额std_10_排名']
#     df['排名'] = df.groupby('交易日期')['因子'].rank()
#     df = df[df['排名'] <= select_stock_num]
#
#     return session_id, df


def financial_report_strategy1(pick_from_df, select_stock_num):
    """
    https://bbs.quantclass.cn/thread/2553
    财报严选财务策略1
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100004

    pick_from_df = rule_out_stocks_global(pick_from_df)

    df = pick_from_df
    # 筛选
    df = df[df['经营活动产生的现金流量净额'] / df['净利润'] > 1]

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    # df['量价相关系数_10_排名'] = df.groupby('交易日期')['量价相关系数_10'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


def small_capital_financial_strategy1(pick_from_df, select_stock_num):
    """
    小市值策略_基本面优化1
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100004

    pick_from_df = rule_out_stocks_global(pick_from_df)

    df = pick_from_df

    # 计算ROE百分比排名，去除ROE较差的20%的股票
    df['ROE排名'] = df.groupby('交易日期')['ROE_单季'].rank(ascending=False, method='min', pct=True)
    df = df[df['ROE排名'] < 0.8]
    # 计算总市值排名
    df['总市值排名'] = df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True, method='min')
    # 计算归母净利润同比增速排名
    df['归母净利润同比增速排名'] = df.groupby('交易日期')['归母净利润同比增速_60'].rank(ascending=False, method='min')
    # 计算复合因子
    df['复合因子'] = df['总市值排名'] + df['归母净利润同比增速排名']

    df['排名'] = df.groupby('交易日期')['复合因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


# def combined_financial_filters_strategy1(pick_from_df, select_stock_num):
#     """
#     https://bbs.quantclass.cn/thread/2586
#     筛选条件组合遍历策略
#     :param buy_amount: 最大仓位
#     :param pick_from_df: 用于选股的数据
#     :return:
#     """
#     session_id = 100004
#
#     if not Second_Board_available:
#         pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
#     if not STAR_Market_available:
#         pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
#
#     df = pick_from_df
#
#     cond1 = df['经营活动产生的现金流量净额'] > 0
#     cond2 = df['营业收入'] > 0
#     cond3 = df['经营活动产生的现金流量净额'] > 100000
#
#     df = df[cond1 & cond2 & cond3]
#
#     factor_list = [('成交额std_10', True, 1), ('成交额_10', True, 0.1),
#                    ('成交额std_20', True, 0.1), ('成交额std_5', True, 0.2),
#                    ('营业收入_单季环比', False, 0.4), ('bias_5', True, 0.1),
#                    ('现金及现金等价物净增加额', True, 0.1), ('应收票据及应收账款', False, 0.1)]
#
#     for (factor, ascending, rate) in factor_list:
#         df[factor + '_排名'] = df.groupby('交易日期')[factor].rank(ascending=ascending)
#
#     df['因子'] = 0
#     for (factor, ascending, rate) in factor_list:
#         df['因子'] += df[factor + '_排名'] * rate
#
#     # 选股
#     df['排名'] = df.groupby('交易日期')['因子'].rank()
#     df = df[df['排名'] <= select_stock_num]
#
#     return session_id, df


def reinforced_factors_strategy(pick_from_df, select_stock_num):
    """
    https://bbs.quantclass.cn/thread/2513
    因子遍历增强策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100004

    pick_from_df = rule_out_stocks_global(pick_from_df)

    df = pick_from_df

    df = df[df['经营活动产生的现金流量净额'] > 0]

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


def ROC_turnover_rate_strategy(pick_from_df, select_stock_num):
    """
    https://bbs.quantclass.cn/thread/12907
    ROC换手率策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100004

    pick_from_df = rule_out_stocks_global(pick_from_df)

    df = pick_from_df

    # 筛选
    df = df[df['bias_5'] >= -0.1]

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['ROC_5_排名'] = df.groupby('交易日期')['ROC_5'].rank()
    df['换手率mean_10_排名'] = df.groupby('交易日期')['换手率mean_10'].rank()

    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['ROC_5_排名'] + df['换手率mean_10_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


# def brutal_combination_strategy(pick_from_df, select_stock_num):
#     """
#     https://bbs.quantclass.cn/thread/2864
#     暴力组合策略
#     :param buy_amount: 最大仓位
#     :param pick_from_df: 用于选股的数据
#     :return:
#     """
#     session_id = 100004
#
#     if not Second_Board_available:
#         pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
#     if not STAR_Market_available:
#         pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
#
#     df = pick_from_df
#
#     n = 9
#     df['mean_%s' % n] = df['收盘价'].rolling(n).mean()
#     df['斜率21_%s' % n] = df['mean_%s' % n].rolling(n).apply(
#         lambda x: np.polyfit([1, 2, 3, 4, 5, 6, 7, 8, 9], x.tolist(), deg=1)[0])
#     df['斜率21_9_排名'] = -df.groupby('交易日期')['斜率21_9'].rank(pct=True)
#
#     # 计算各因子归一化值
#     factor_list = ['成交额std_10', '资产总计', '涨跌幅std_20', '经营活动产生的现金流量净额_ttm', '净利润_ttm同比', '营业总收入_ttm']
#
#     for factor in factor_list:
#         df[factor + '_排名'] = df.groupby('交易日期')[factor].rank()  # 进行归一化
#     # 选股
#     df['因子'] = df['成交额std_10_排名'] * df['资产总计_排名'] + df['涨跌幅std_20_排名'] - df['现金流量净额(经营活动)(ttm)_排名'] + df[
#         '净利润(ttm)_同比_排名'] + df['营业总收入(ttm)_排名'] + df['斜率21_9_排名']
#     df['排名'] = df.groupby('交易日期')['因子'].rank()
#     df = df[df['排名'] <= select_stock_num]
#
#     return session_id, df


def rnd_expense_strategy(pick_from_df, select_stock_num):
    """
    https://bbs.quantclass.cn/thread/2437
    研发费用策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100004

    pick_from_df = rule_out_stocks_global(pick_from_df)

    df = pick_from_df

    # 排序
    df['成交额std_10_排名'] = df.groupby('交易日期')['成交额std_10'].rank()
    df['研发费用_排名'] = df.groupby('交易日期')['研发费用'].rank()
    # 选股
    df['因子'] = df['成交额std_10_排名'] + df['研发费用_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df


# def bollinger_income_strategy(pick_from_df, select_stock_num):
#     """
#     https://bbs.quantclass.cn/thread/2993
#     营业总收入_环比布林组合策略
#     :param buy_amount: 最大仓位
#     :param pick_from_df: 用于选股的数据
#     :return:
#     """
#     session_id = 100004
#
#     pick_from_df = rule_out_stocks_global(pick_from_df)
#
#     df = pick_from_df
#
#     # 计算指标
#     df['研发投入比'] = df['研发费用_ttm'] / df['营业总收入_ttm']
#
#     # 筛选
#     cond1 = df['营业总收入_单季环比'] >= 0.18
#     cond2 = df['营业收入'] / df['营业总收入'] > 0.90
#
#     # 进行归一化
#     factor_list = ['成交额std_10', '研发投入比']
#     for factor in factor_list:
#         df[factor + '_排名'] = df.groupby('交易日期')[factor].rank(pct=True)
#     df = df[cond1 & cond2]
#
#     # 布林优化
#     n = 9
#     df['close'] = (df['bias_5'] + 1) * df['均线_5']  # 求 df['收盘价_复权']
#     df['median'] = df['close'].rolling(n, min_periods=1).mean()
#     df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
#     df['z_score'] = abs(df['close'] - df['median']) / df['std']
#     df['m'] = df['z_score'].rolling(window=n).max().shift()
#     df['upper'] = df['median'] + df['std'] * df['m']
#     df['lower'] = df['median'] - df['std'] * df['m']
#     df = df[df['close'] > df['lower']]
#
#     # 选股
#     df['因子'] = df['成交额std_10_排名'] * 2 - df['研发投入比_排名']
#
#     df['排名'] = df.groupby('交易日期')['因子'].rank()
#     df = df[df['排名'] <= select_stock_num]
#
#     return session_id, df
