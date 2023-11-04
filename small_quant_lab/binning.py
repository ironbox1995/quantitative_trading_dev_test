# -*- coding: UTF-8 -*-
import warnings

from Config.global_config import *
from small_quant_lab.factor_industry_exposure_function import *

warnings.filterwarnings('ignore')

period_type = "W"
# 从pickle文件中读取整理好的所有股票数据
df = pd.read_pickle(
    r'{}\data\historical\processed_data\all_stock_data_{}.pkl'.format(project_path, period_type))
df.dropna(subset=['下周期涨跌幅'], inplace=True)
# ===删除下个交易日不交易、开盘涨停的股票，因为这些股票在下个交易日开盘时不能买入。
df = df[df['下日_是否交易'] == 1]
df = df[df['下日_开盘涨停'] == False]
df = df[df['下日_是否ST'] == False]
df = df[df['下日_是否退市'] == False]
# df['涨跌幅排名'] = df.groupby('交易日期')['下周期涨跌幅'].rank(ascending=True)

factor_li = ['成交额', '流通市值（万元）', '总市值 （万元）', '上市至今交易天数', '总股本 （万股）', '流通股本 （万股）', '自由流通股本 （万）', '成交量', '量比', '市盈率', '市盈率TTM', '市净率', '市销率', '市销率TTM', '股息率（%）', '股息率TTM（%）', '前日成交额', '换手率', '前日换手率', '5日均线', '20日均线', 'bias_5', '5日涨跌幅环比变化', 'bias_20', '20日涨跌幅环比变化', '20日涨跌幅', 'alpha95', '账面市值比', 'VWAP', 'A/D', 'TYP', 'AMOV', '均线_5', '涨跌幅_5', '涨跌幅std_5', '收盘价std_5', '成交额std_5', '成交额_5', '振幅_5', '量价相关性_5', '换手率mean_5', '非流动性_5', 'MTM_5', 'OCS_5', 'ROC_5', 'WR_5', 'CYF_5', 'EMV_5', 'VPT_5', 'JS_5', 'AMV_5', '均线_10', 'bias_10', '涨跌幅_10', '涨跌幅std_10', '收盘价std_10', '成交额std_10', '成交额_10', '振幅_10', '量价相关性_10', '换手率mean_10', '非流动性_10', 'MTM_10', 'OCS_10', 'ROC_10', 'WR_10', 'CYF_10', 'EMV_10', 'VPT_10', 'JS_10', 'AMV_10', '均线_20', '涨跌幅_20', '涨跌幅std_20', '收盘价std_20', '成交额std_20', '成交额_20', '振幅_20', '量价相关性_20', '换手率mean_20', '非流动性_20', 'MTM_20', 'OCS_20', 'ROC_20', 'WR_20', 'CYF_20', 'EMV_20', 'VPT_20', 'JS_20', 'AMV_20', 'cro_5', 'cpo_5', 'cro_10', 'cpo_10', 'cro_15', 'cpo_15', 'cro_20', 'cpo_20', '单日振幅20日均值', '20日振幅', '周期内成交额', '周期内最后交易日流通市值', '周期换手率', '本周期涨跌幅']

# 小市值normalIC
df = df[df['总市值 （万元）'] < 300000]
if not Second_Board_available:
    df = df[df['市场类型'] != '创业板']
if not STAR_Market_available:
    df = df[df['市场类型'] != '科创板']
start_date = "20100101"
end_date = '20230930'

# start_date = "20220101"
# end_date = '20230701'

for factor in factor_li:
    try:
        curve_cap_factor_boxplot1(df, factor, start_date, end_date)
    except:
        print(f"计算{factor}分箱失败！")
