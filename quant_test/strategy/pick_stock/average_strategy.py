from strategy.strategy_utils import *


def average_20_day_strategy(pick_from_df, select_stock_num):
    """
    单均线策略20日
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100016

    pick_from_df = rule_out_stocks_global(pick_from_df, select_stock_num)

    # pick_from_df = pick_from_df[pick_from_df['行业'].isin(['银行'])]
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['bias_20'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df


def average_5_day_strategy(pick_from_df, select_stock_num):
    """
    单均线策略5日
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100017

    pick_from_df = rule_out_stocks_global(pick_from_df, select_stock_num)

    # pick_from_df = pick_from_df[pick_from_df['行业'].isin(['银行'])]  # 有了行业选择之后其实会方便很多
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['bias_5'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df
