# 思路：使用Q学习中的eps-greedy方法并联策略，尝试寻找最优解，为防止所有策略同时失效的可能性，应加入空仓策略
# 所有策略均选择未择时的策略
from backtest.Evaluate import *
from data.processing.Functions import *
from get_strategy_function import get_strategy_function
from backtest.repick_time import *
from backtest.latest_result import back_test_latest_result
from back_test_config import *
import warnings
from utils_global.dingding_message import *

import datetime


def create_empty_strategy():
    # 导入指数数据
    index_data = import_index_data(
        r"F:\quantitative_trading_dev_test\quant_test\data\historical\tushare_index_data\000001.SH.csv"
        , back_trader_start=date_start, back_trader_end=date_end)
    # 创造空的事件周期表，用于填充不选股的周期
    empty_df = create_empty_data(index_data, 'W')
    empty_df['Q'] = 0
    return empty_df


def load_all_strategies():
    period_type = 'W'
    select_stock_num = 3
    strategy_name = 'Q学习并联策略'

    strategy_dct = {}

    for strategy_name in strategy_li:
        select_stock = back_test_latest_result(strategy_name, select_stock_num, period_type, pick_time_mtd="")  # 无择时
        strategy_dct[strategy_name] = select_stock

    strategy_dct['空仓策略'] = create_empty_strategy()

    # 以下代码来自ChatGPT，需要检查
    # TODO：实现epsilon-greedy
    # 使用pd.concat将它们合并成一个新的DataFrame，并保留原始的行索引
    merged_df = pd.concat(strategy_dct.values())

    # 使用groupby方法按原始行索引聚合找到C列的最大值，同时保留其他列的信息
    select_stock = merged_df.groupby(level=0).apply(lambda group: group.loc[group['Q'].idxmax()])
    # 以上代码来自ChatGPT，需要检查

    # 常量设置
    c_rate = 1 / 10000  # 手续费 这里与之前不同
    t_rate = 1 / 1000  # 印花税

    # 导入指数数据
    index_data = import_index_data(
        r"F:\quantitative_trading_dev_test\quant_test\data\historical\tushare_index_data\000001.SH.csv"
        , back_trader_start=date_start, back_trader_end=date_end)

    # 创造空的事件周期表，用于填充不选股的周期
    empty_df = create_empty_data(index_data, period_type)

    # 计算整体资金曲线
    select_stock.reset_index(inplace=True)
    select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
    select_stock.set_index('交易日期', inplace=True)
    empty_df.update(select_stock)
    empty_df['资金曲线'] = select_stock['资金曲线']
    select_stock = empty_df
    select_stock.reset_index(inplace=True, drop=False)

    select_stock.to_csv(
        r"F:\quantitative_trading_dev_test\quant_test\backtest\result_record\select_stock_{}_{}_选{}_{}-{}_{}.csv"
        .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd), encoding='gbk')

    # ===计算选中股票每天的资金曲线
    # 计算每日资金曲线
    equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                      how='left', sort=True)  # 将选股结果和大盘指数合并
    # equity.to_csv("equity.csv", encoding='utf_8_sig')

    equity['持有股票代码'] = equity['买入股票代码'].shift()
    equity['持有股票代码'].fillna(method='ffill', inplace=True)
    equity.dropna(subset=['持有股票代码'], inplace=True)
    # del equity['买入股票代码']

    equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()  # 累加是没错的
    equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
    equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()

    equity.to_csv(r"F:\quantitative_trading_dev_test\quant_test\backtest\result_record\equity_{}_{}_选{}_{}-{}_{}.csv"
                  .format(strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd),
                  encoding='gbk')
