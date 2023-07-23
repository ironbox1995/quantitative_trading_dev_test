import pandas as pd


def revert_strategy(pick_from_df, select_stock_num):
    """
    反转策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100002  # 代表选股策略中的第二个

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['涨跌幅'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    # buy_stock_list = df["股票代码"].values.tolist()

    return session_id, df
