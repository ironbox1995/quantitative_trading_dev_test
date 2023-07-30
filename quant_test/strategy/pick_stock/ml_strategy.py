from strategy.pick_stock.machine_learning.predict import *


def random_forest_classify_strategy(pick_from_df, select_stock_num, period_type):
    """
    使用随机森林分类，并选出市值最小的3只股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200001

    pick_from_df = ML_model_predictor(pick_from_df, period_type, "random_forest_classify")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] == "1"]
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]

    return session_id, df


def random_forest_regress_strategy(pick_from_df, select_stock_num, period_type):
    """
    选择随机森林预测的下周期增长最高的3个股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200002

    pick_from_df = ML_model_predictor(pick_from_df, period_type, "random_forest_regress")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] >= 0.002]  # 至少要能覆盖印花税和手续费
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['机器学习预测值'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df


def svm_classify_strategy(pick_from_df, select_stock_num, period_type):
    """
    使用SVM分类，并选出市值最小的3只股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200003

    pick_from_df = ML_model_predictor(pick_from_df, period_type, "SVC_classify")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] == "1"]
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df


def svm_regress_strategy(pick_from_df, select_stock_num, period_type):
    """
    选择SVM预测的下周期增长最高的3个股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200004

    pick_from_df = ML_model_predictor(pick_from_df, period_type, "SVR_regress")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] >= 0.002]  # 至少要能覆盖印花税和手续费
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['机器学习预测值'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df


def fcn_classify_strategy(pick_from_df, select_stock_num, period_type):
    """
    使用SVM分类，并选出市值最小的3只股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200005

    pick_from_df = DL_model_classify_predictor(pick_from_df, period_type, "FCN_classify")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] == "1"]
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['总市值 （万元）'].rank(ascending=True)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df


def fcn_regress_strategy(pick_from_df, select_stock_num, period_type):
    """
    选择SVM预测的下周期增长最高的3个股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200006

    pick_from_df = DL_model_regress_predictor(pick_from_df, period_type, "FCN_regress")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] >= 0.002]  # 至少要能覆盖印花税和手续费
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['机器学习预测值'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df


def lstm_strategy(pick_from_df, select_stock_num, period_type):
    pass


def rl_strategy(pick_from_df, select_stock_num, period_type):
    pass
