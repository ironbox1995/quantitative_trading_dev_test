from strategy.strategy_config import *


def consistent_strategy(pick_from_df, select_stock_num):
    """
    惯性策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 100004

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']

    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['涨跌幅'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    # buy_stock_list = df["股票代码"].values.tolist()

    return session_id, df
