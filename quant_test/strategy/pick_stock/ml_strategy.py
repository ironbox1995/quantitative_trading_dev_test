from strategy.pick_stock.machine_learning.predict import *
from utils_global.global_config import *


def random_forest_classify_strategy(pick_from_df, select_stock_num, period_type):
    """
    使用随机森林分类，并选出市值最小的3只股票
    :param pick_from_df:
    :param select_stock_num:
    :param period_type:
    :return:
    """
    session_id = 200001

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

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

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    pick_from_df = DL_model_regress_predictor(pick_from_df, period_type, "FCN_regress")
    pick_from_df = pick_from_df[pick_from_df["机器学习预测值"] >= 0.002]  # 至少要能覆盖印花税和手续费
    pick_from_df['排名'] = pick_from_df.groupby('交易日期')['机器学习预测值'].rank(ascending=False)
    df = pick_from_df[pick_from_df['排名'] <= select_stock_num]
    return session_id, df


def lstm_strategy(pick_from_df, select_stock_num, period_type):

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    pass


def rl_strategy(pick_from_df, select_stock_num, period_type):

    if not Second_Board_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '创业板']
    if not STAR_Market_available:
        pick_from_df = pick_from_df[pick_from_df['市场类型'] != '科创板']
    if use_black_list:
        pick_from_df = pick_from_df[~pick_from_df['股票代码'].isin(black_list)]  # 使用isin()函数和~操作符来排除包含这些值的行

    pass
