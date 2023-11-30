from predictor.backup.machine_learning.predict import *
from strategy.strategy_utils import *


def fcn_regress_strategy(pick_from_df, select_stock_num, period_type):
    """
    选择FCN预测的下周期增长最高的3个股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200006

    pick_from_df = rule_out_stocks_global(pick_from_df)

    pick_from_df = DL_model_regress_predictor(pick_from_df, period_type, "FCN_regress")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] >= 0.002]  # 至少要能覆盖印花税和手续费
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['机器学习预测值'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df
