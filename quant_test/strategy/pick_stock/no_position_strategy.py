import pandas as pd


def no_position_strategy(pick_from_df, select_stock_num):
    """
    空仓策略
    :param pick_from_df:
    :param select_stock_num:
    :return:
    """
    session_id = 100000
    empty_df = pd.DataFrame(columns=pick_from_df.columns, index=pick_from_df.index)
    return session_id, empty_df
