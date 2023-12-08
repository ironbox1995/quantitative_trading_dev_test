from strategy.strategy_utils import *


def amplitude_20_day_strategy(pick_from_df, select_stock_num):
    """
    20日振幅策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100005

    pick_from_df = rule_out_stocks_global(pick_from_df, select_stock_num)

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['20日振幅'].rank(ascending=True)  # 平均振幅小的好
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    # buy_stock_list = df["股票代码"].values.tolist()

    return session_id, df


def one_day_amplitude_20_day_average_strategy(pick_from_df, select_stock_num):
    """
    20日振幅策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100006

    pick_from_df = rule_out_stocks_global(pick_from_df, select_stock_num)

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['单日振幅20日均值'].rank(ascending=True)  # 平均振幅小的好
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    # buy_stock_list = df["股票代码"].values.tolist()

    return session_id, df
