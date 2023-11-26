import pandas as pd


def no_position_strategy(pick_from_df, select_stock_num):
    """
    空仓策略 （使用大市值策略代替，回测时将结果处理为空仓）
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100000
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, pick_from_df
