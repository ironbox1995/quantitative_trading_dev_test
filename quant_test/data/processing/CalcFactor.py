"""
数据整理需要计算的因子脚本，可以在这里修改/添加别的因子计算
"""


def cal_tech_factor(df, extra_agg_dict):
    """
    计算量价因子
    :param df:
    :param extra_agg_dict:
    :return:
    """

    # =计算前日成交额
    df['前日成交额'] = df['成交额']
    extra_agg_dict['前日成交额'] = 'last'

    # =计算换手率
    df['换手率'] = df['换手率（%）'] / 100
    extra_agg_dict['换手率'] = 'mean'

    # 计算前日换手率
    df['前日换手率'] = df['换手率']
    extra_agg_dict['前日换手率'] = 'last'

    # =计算5日均线
    df['5日均线'] = df['收盘价_复权'].rolling(5).mean()
    extra_agg_dict['5日均线'] = 'last'

    # =计算20日均线
    df['20日均线'] = df['收盘价_复权'].rolling(20).mean()
    extra_agg_dict['20日均线'] = 'last'

    # =计算bias_5
    df['bias_5'] = df['收盘价_复权'] / df['5日均线'] - 1
    extra_agg_dict['bias_5'] = 'last'

    # =计算5日累计涨跌幅
    df['5日涨跌幅环比变化'] = df['涨跌幅'].pct_change(5)
    extra_agg_dict['5日涨跌幅环比变化'] = 'last'

    # =计算bias_20
    df['bias_20'] = df['收盘价_复权'] / df['20日均线'] - 1
    extra_agg_dict['bias_20'] = 'last'

    # =计算20日累计涨跌幅
    df['20日涨跌幅环比变化'] = df['涨跌幅'].pct_change(20)
    extra_agg_dict['20日涨跌幅环比变化'] = 'last'

    # df['量价相关性'] = df['收盘价_复权'].rolling(10).corr(df['换手率'])
    # extra_agg_dict['量价相关性'] = 'last'

    # 20日涨跌幅：后复权价的20日环比变化
    df["20日涨跌幅"] = df['收盘价_复权'].pct_change(20)
    extra_agg_dict['20日涨跌幅'] = 'last'

    # =alpha95因子(五天成交额的标准差)
    df['alpha95'] = df['成交额'].rolling(5).std()
    extra_agg_dict['alpha95'] = 'last'

    # bm：book-to-market ratio 账面市值比
    df['账面市值比'] = 1 / df['市净率']
    extra_agg_dict['账面市值比'] = 'last'

    try:
        df = cal_tech_factor_all(df, extra_agg_dict)
    except Exception as e:
        print("计算全量指标失败，请检查！")
        print(e)

    return df


# 计算量价因子函数
def cal_tech_factor_all(df, extra_agg_dict):
    """
    计算量价因子
    :param df:
    :param extra_agg_dict:
    :return:
    """
    # =计算均价
    df['VWAP'] = df['成交额'] / df['成交量']
    extra_agg_dict['VWAP'] = 'last'

    df['非流动性'] = abs(df['涨跌幅'] / df['成交额']) * 100000000
    extra_agg_dict['非流动性'] = 'mean'

    # ===计算MACD指标
    # df['DIF'] = df['收盘价_复权'].ewm(alpha=2 / 13, adjust=False).mean() - df['收盘价_复权'].ewm(alpha=2 / 27, adjust=False).mean()
    # df['DEA'] = df['DIF'].ewm(alpha=2 / 10, adjust=False).mean()
    # df.loc[df['DIF'] / df['DEA'] > 1, '看涨'] = True
    # df['MACD_Hist'] = abs(df['DEA'] - df['DIF'])
    # df['L97'] = 2 * (df['DIF'] - df['DIF'].ewm(alpha=2 / 10, adjust=False).mean())
    # extra_agg_dict['看涨'] = 'last'
    # extra_agg_dict['MACD_Hist'] = 'last'
    # extra_agg_dict['L97'] = 'last'
    # low_list = df['最低价_复权'].rolling(9, min_periods=1).min()
    # high_list = df['最高价_复权'].rolling(9, min_periods=1).max()
    # rsv = (df['收盘价_复权'] - low_list) / (high_list - low_list) * 100
    # df['K'] = rsv.ewm(com=2).mean()
    # df['D'] = df['K'].ewm(com=2).mean()
    # df['J'] = 3 * df['K'] - 2 * df['D']
    # extra_agg_dict['K'] = 'last'
    # extra_agg_dict['D'] = 'last'
    # extra_agg_dict['J'] = 'last'
    # df['上影线'] = (df['最高价_复权'] - df['收盘价_复权']) / df['收盘价_复权']
    # df['下影线'] = (df['收盘价_复权'] - df['最低价_复权']) / df['收盘价_复权']
    # extra_agg_dict['上影线'] = 'last'
    # extra_agg_dict['下影线'] = 'last'

    df['A/D'] = (df['收盘价_复权'] - df['开盘价_复权']) / (df['最高价_复权'] - df['最低价_复权']) * df['成交量']
    extra_agg_dict['A/D'] = 'last'
    df['TYP'] = (df['最高价_复权'] + df['收盘价_复权'] + df['最低价_复权']) / 3
    extra_agg_dict['TYP'] = 'last'
    df['AMOV'] = df['成交量'] * ((df['开盘价_复权'] + df['收盘价_复权']) / 2)
    extra_agg_dict['AMOV'] = 'last'

    # 有多个窗口期的量价指标
    # n_list = [3, 5, 10, 15, 20, 30, 40, 60]
    n_list = [5, 10, 20]
    for n in n_list:
        # ===计算均线
        df['均线_%s' % n] = df['收盘价_复权'].rolling(n, min_periods=1).mean()
        extra_agg_dict['均线_%s' % n] = 'last'

        # ===计算bias
        df['bias_%s' % n] = df['收盘价_复权'] / df['均线_%s' % n] - 1
        extra_agg_dict['bias_%s' % n] = 'last'

        # ===计算涨跌幅
        df['涨跌幅_%d' % n] = df['收盘价_复权'].pct_change(n)
        extra_agg_dict['涨跌幅_%d' % n] = 'last'

        # ===计算涨跌幅标准差
        df['涨跌幅std_%s' % n] = df['涨跌幅'].rolling(n, min_periods=1).std()
        extra_agg_dict['涨跌幅std_%d' % n] = 'last'

        # ===收盘价std
        df['收盘价std_%s' % n] = df['收盘价_复权'].rolling(n, min_periods=1).std()
        extra_agg_dict['收盘价std_%s' % n] = 'last'

        # ===计算成交额标准差
        df['成交额std_%d' % n] = df['成交额'].rolling(n, min_periods=1).std(ddof=0)
        extra_agg_dict['成交额std_%d' % n] = 'last'

        # ===计算成交额sum
        df['成交额_%d' % n] = df['成交额'].rolling(n, min_periods=1).sum()
        extra_agg_dict['成交额_%d' % n] = 'last'

        # ===计算振幅
        df['振幅_%d' % n] = df['最高价_复权'].rolling(n, min_periods=1).max() / df['最低价_复权'].rolling(n, min_periods=1).min()
        extra_agg_dict['振幅_%d' % n] = 'last'

        # =计算量价相关因子
        df['量价相关性_%d' % n] = df['收盘价_复权'].rolling(n, min_periods=1).corr(df['换手率'])
        extra_agg_dict['量价相关性_%d' % n] = 'last'
        #
        # ===计算换手率均值
        df['换手率mean_%d' % n] = df['换手率'].rolling(n, min_periods=1).mean()
        extra_agg_dict['换手率mean_%d' % n] = 'last'

        # ===非流动性
        df['非流动性_%d' % n] = abs(df['涨跌幅_%d' % n] / df['成交额_%d' % n]) * 100000000
        extra_agg_dict['非流动性_%d' % n] = 'mean'

        # ===MTM
        df['MTM_%d' % n] = df['收盘价_复权'] - df['收盘价_复权'].shift(n)
        extra_agg_dict['MTM_%d' % n] = 'last'

        # # ===OCS
        df['OCS_%d' % n] = 100 * (df['收盘价_复权'] - df['收盘价_复权'].rolling(n, min_periods=1).mean())
        extra_agg_dict['OCS_%d' % n] = 'last'

        # # ===ROC
        df['ROC_%d' % n] = 100 * ((df['收盘价_复权'] - df['收盘价_复权'].shift(n)) / df['收盘价_复权'].shift(n))
        extra_agg_dict['ROC_%d' % n] = 'last'

        # ===WR
        df['WR_%d' % n] = 100 * ((df['最高价_复权'].rolling(n, min_periods=1).max() - df['收盘价_复权'])
                                 / (df['最高价_复权'].rolling(n, min_periods=1).max() - df['最低价_复权'].rolling(n, min_periods=1).min()))
        extra_agg_dict['WR_%d' % n] = 'last'

        # ===CYF
        df['CYF_%d' % n] = 100 - 100 / (1 + df['换手率'].ewm(alpha=2 / (n + 1), adjust=False).mean())
        extra_agg_dict['CYF_%d' % n] = 'last'

        # ===EMV
        df['EMV_%d' % n] = ((100 * ((df['最高价_复权'] + df['最低价_复权']) - (df['最高价_复权'].shift() + df['最低价_复权'].shift()))
                             / (df['最高价_复权'] + df['最低价_复权'])) * df['成交量'].rolling(n, min_periods=1).mean() *
                            ((df['最高价_复权'] - df['最低价_复权']) / df['最高价_复权'].rolling(n).mean() - df['最低价_复权'].rolling(n).mean())).rolling(n, min_periods=1).mean()
        extra_agg_dict['EMV_%d' % n] = 'last'

        # ===VPT
        df['VPT_%d' % n] = (df['成交量'] * ((df['收盘价_复权'] - df['收盘价_复权'].shift()) / df['收盘价_复权'].shift())).rolling(n, min_periods=1).sum()
        extra_agg_dict['VPT_%d' % n] = 'last'

        # ===JS
        df['JS_%d' % n] = (df['收盘价_复权'] - df['收盘价_复权'].shift(n)) / (n * df['收盘价_复权'].shift(n)) * 100
        extra_agg_dict['JS_%d' % n] = 'last'

        # ===AMV
        df['AMV_%d' % n] = df['AMOV'].rolling(n, min_periods=1).sum() / df['成交量'].rolling(n, min_periods=1).sum()
        extra_agg_dict['AMV_%d' % n] = 'last'

    # 量价时序指标
    # t_list = [10, 15, 20, 40, 60, 120, 240]
    t_list = [5, 10, 15, 20]
    for t in t_list:
        # ===计算cro
        df['cro_%s' % t] = df['涨跌幅'].rolling(t, min_periods=1).corr(df['上市至今交易天数'])
        extra_agg_dict['cro_%s' % t] = 'last'
        # ===计算cpo
        df['cpo_%s' % t] = df['收盘价_复权'].rolling(t, min_periods=1).corr(df['上市至今交易天数'])
        extra_agg_dict['cpo_%s' % t] = 'last'

    return df


def calc_fin_factor(df, extra_agg_dict):
    """
    计算财务因子
    :param df:              原始数据
    :param finance_df:      财务数据
    :param extra_agg_dict:  resample需要用到的
    :return:
    """

    # ====计算常规的财务指标
    # ===计算归母PE
    # 归母PE = 总市值 / 归母净利润(ttm)
    # print(df.columns)
    df['归母PE(ttm)'] = df['总市值 （万元）'] / df['归母净利润']
    extra_agg_dict['归母PE(ttm)'] = 'last'

    # ===计算归母ROE
    # 归母ROE(ttm) = 归母净利润(ttm) / 归属于母公司股东权益合计
    df['归母ROE(ttm)'] = df['归母净利润_ttm'] / df['归母所有者权益合计']
    extra_agg_dict['归母ROE(ttm)'] = 'last'

    # ===计算毛利率ttm
    # 毛利率(ttm) = 营业总收入_ttm / 营业总成本_ttm - 1
    df['毛利率(ttm)'] = df['营业总收入_ttm'] / df['营业总成本_ttm'] - 1
    extra_agg_dict['毛利率(ttm)'] = 'last'

    # SP指标
    df['SP'] = df['营业总收入_单季'] / df['总市值 （万元）']
    extra_agg_dict['SP'] = 'last'

    # ===计算企业倍数指标：EV2 / EBITDA
    """
    EV2 = 总市值 + 有息负债 - 货币资金,
    # EBITDA税息折旧及摊销前利润
    EBITDA = 营业总收入-营业税金及附加-营业成本+利息支出+手续费及佣金支出+销售费用+管理费用+研发费用+坏账损失+存货跌价损失+固定资产折旧、油气资产折耗、生产性生物资产折旧+无形资产摊销+长期待摊费用摊销+其他收益
    """
    # 有息负债 = 短期借款 + 长期借款 + 应付债券 + 一年内到期的非流动负债
    df['有息负债'] = df[['短期借款', '长期借款', '应付债券', '一年内到期的非流动负债']].sum(axis=1)
    # 计算EV2
    df['EV2'] = df['总市值 （万元）'] + df['有息负债'] - df['货币资金'].fillna(0)
    extra_agg_dict['EV2'] = 'last'

    # 计算EBITDA
    # 坏账损失 字段无法直接从财报中获取，暂去除不计
    df['EBITDA'] = df[[
        '营业总收入', '负债应付利息', '应付手续费及佣金',
        '销售费用', '管理费用', '研发费用', '资产减值损失',
        '固定资产折旧、油气资产折耗、生产性物资折旧', '无形资产摊销', '长期待摊费用摊销',
        '其他综合利益', '流动负债合计', '非流动负债合计'
    ]].sum(axis=1) - df[['税金及附加', '营业成本']].sum(axis=1)
    extra_agg_dict['EBITDA'] = 'last'

    # 计算企业倍数
    df['企业倍数'] = df['EV2'] / df['EBITDA']
    extra_agg_dict['企业倍数'] = 'last'

    # ===计算现金流负债比
    # 现金流负债比 = 现金流量净额(经营活动) / 总负债(流动负债合计 + 非流动负债合计)
    df['现金流负债比'] = df['经营活动产生的现金流量净额'] / (df['流动负债合计'] + df['非流动负债合计'])
    extra_agg_dict['现金流负债比'] = 'last'

    # =归母净利润同比增速 相较于60个交易日前的变化
    df['归母净利润_单季同比_60'] = df['归母净利润_单季同比'].shift(60)
    df['归母净利润同比增速_60'] = df['归母净利润_单季同比'] - df['归母净利润_单季同比_60']
    extra_agg_dict['归母净利润同比增速_60'] = 'last'

    # ===计算单季度ROE
    df['ROE_单季'] = df['归母净利润_单季'] / df['归母所有者权益合计']
    extra_agg_dict['ROE_单季'] = 'last'

    return extra_agg_dict, df
