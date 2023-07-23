import pandas as pd


def no_position_strategy(pick_from_df, select_stock_num):
    """
    空仓策略
    :param pick_from_df:
    :param select_stock_num:
    TODO: 此策略有问题待修正
    :return:
    """
    session_id = 100000
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=False)
    pick_from_df = pick_from_df[pick_from_df['排名'] <= -1]  # 选出排名为负数的股票
    return session_id, pick_from_df
