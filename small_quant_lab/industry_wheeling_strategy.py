from utils_global.global_config import *


def industry_wheeling_strategy1(pick_from_df, select_stock_num):
    """
    行业轮动策略
    :param buy_amount: 最大仓位
    :param pick_from_df: 用于选股的数据
    :return:
    """
    session_id = 200001

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    df = pick_from_df

    # 选股
    df['因子'] = df['成交额std_10_排名']
    df['排名'] = df.groupby('交易日期')['因子'].rank()
    df = df[df['排名'] <= select_stock_num]

    return session_id, df