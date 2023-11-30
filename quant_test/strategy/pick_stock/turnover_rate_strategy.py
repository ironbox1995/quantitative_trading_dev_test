from strategy.strategy_utils import *


def turnover_rate_strategy(pick_from_df, select_stock_num):
    """
    换手率策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100008

    pick_from_df = rule_out_stocks_global(pick_from_df)

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['换手率（%）'].rank(ascending=True)  # 选择换手率低的：ascending=True
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    # buy_stock_list = df["股票代码"].values.tolist()

    return session_id, df


def volume_turnover_strategy1(pick_from_df, select_stock_num):
    """
    成交额换手率策略1
    https://bbs.quantclass.cn/thread/12907
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100020

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
