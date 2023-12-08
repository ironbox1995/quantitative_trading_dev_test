from Config.global_config import *
from Config.trade_config import *


def rule_out_stocks_global(pick_from_df, select_stock_num=3):
    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    # 避免价格太高导致下单失败
    stock_price_limit = total_position / (100*select_stock_num*1.1)
    pick_from_df = pick_from_df[pick_from_df['收盘价'] <= stock_price_limit]

    return pick_from_df
