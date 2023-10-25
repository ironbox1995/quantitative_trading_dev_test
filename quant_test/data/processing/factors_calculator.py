from data.processing.CalcFactor import *
from data.processing.Function_fin import *
from data.processing.Functions import *
from data.processing.reformat_utils import *
from data.processing.data_config import *
from utils_global.global_config import *

import traceback


def daily_factor_calculator(code, df, index_data):
    """
    基于每日数据的因子计算
    :param price_volume: 是否计算量价因子
    :param df: 待选股数据
    :return:
    """
    # 计算换手率
    # df['换手率'] = df['成交额'] / df['流通市值']
    # 计算涨跌幅
    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1

    # 计算复权因子：假设你一开始有1元钱，投资到这个股票，最终会变成多少钱。
    df = cal_fuquan_price(df, fuquan_type='后复权')

    df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1  # 为之后开盘买入做好准备

    # 计算交易天数
    df['上市至今交易天数'] = df.index + 1

    # 计算涨跌停价格
    df = cal_zdt_price(df)

    # 需要额外保存的字段
    extra_fill_0_list = []  # 在和上证指数合并时使用。
    extra_agg_dict = {}  # 在转换周期时使用。
    # 将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
    df = merge_with_index_data(df, index_data, extra_fill_0_list)
    if df.empty:
        return pd.DataFrame()

    # 计算量价选股因子
    df = cal_tech_factor(df, extra_agg_dict)

    if use_financial_data:
        try:
            # 导入财务数据，并计算相关衍生指标
            # finance_df = import_fin_data(code, finance_data_path)
            finance_df = import_tushare_fin_data(code, finance_data_path)
            if not finance_df.empty:  # 如果数据不为空
                # 计算财务数据：选取需要的字段、计算指定字段的同比、环比、ttm等指标
                finance_df = proceed_fin_data(finance_df, raw_fin_cols, flow_fin_cols, cross_fin_cols, derived_fin_cols, fin_ind_col_final)
                # 财务数据和股票k线数据合并，使用merge_asof
                # 参考文档：https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.merge_asof.html
                df = pd.merge_asof(left=df, right=finance_df, left_on='交易日期', right_on='披露日期',
                                   direction='backward')
            else:  # 如果数据为空
                # 把需要使用的字段都填为空值
                for col in raw_fin_cols + derived_fin_cols + fin_ind_col_final:
                    df[col] = np.nan

            for col in raw_fin_cols:  # 财务数据在周期转换的时候，都是选取最后一天的数据
                extra_agg_dict[col] = 'last'

            for col in derived_fin_cols:
                extra_agg_dict[col] = 'last'

            for col in fin_ind_col_final:
                extra_agg_dict[col] = 'last'

            # 计算财务因子
            extra_agg_dict, df = calc_fin_factor(df, extra_agg_dict)

        except Exception as e:
            print("财务因子处理失败：", e)
            traceback.print_exc()

    # ==== 计算因子
    # 计算基础振幅策略所需列
    df['单日振幅'] = df['最高价'] / df['最低价'] - 1
    df['单日振幅20日均值'] = df['单日振幅'].rolling(20).mean()
    df['20日最大值'] = df['最高价'].rolling(20).max()
    df['20日最小值'] = df['最低价'].rolling(20).min()
    df['20日振幅'] = df['20日最大值'] / df['20日最小值'] - 1
    extra_agg_dict['单日振幅20日均值'] = 'last'
    extra_agg_dict['20日振幅'] = 'last'

    # =计算下个交易的相关情况
    df['下日_是否交易'] = df['是否交易'].shift(-1)
    df['下日_一字涨停'] = df['一字涨停'].shift(-1)
    df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
    df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
    df['下日_是否S'] = df['股票名称'].str.contains('S').shift(-1)
    df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
    df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)

    return extra_agg_dict, df


def periodically_factor_calculator(period_df, df, period_type):
    """
    基于周期数据的因子计算
    :param period_type: W代表周，M代表月
    :param period_df: 按周期数据
    :param df: 每日数据
    :return:
    """
    # 计算必须额外数据
    period_df['交易天数'] = df['是否交易'].resample(period_type).sum()
    period_df['市场交易天数'] = df['股票代码'].resample(period_type).size()
    period_df = period_df[period_df['市场交易天数'] > 0]  # 有的时候整个周期不交易（例如春节、国庆假期），需要将这一周期删除

    # 计算其他因子
    period_df['周期内成交额'] = df['成交额'].resample(period_type).sum()
    period_df['周期内最后交易日流通市值'] = df['流通市值'].resample(period_type).last()
    period_df['周期换手率'] = period_df['周期内成交额']/period_df['周期内最后交易日流通市值']

    # 计算周期资金曲线
    period_df['每天涨跌幅'] = df['涨跌幅'].resample(period_type).apply(lambda x: list(x))

    # 计算振幅
    period_df['单日振幅20日均值'] = df['单日振幅20日均值'].resample(period_type).last()
    period_df['20日振幅'] = df['20日振幅'].resample(period_type).last()

    return period_df
