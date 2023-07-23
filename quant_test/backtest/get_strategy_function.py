from strategy.pick_stock.amplitude_20_strategy import *
from strategy.pick_stock.capitalization_strategy import *
from strategy.pick_stock.consistent_strategy import *
from strategy.pick_stock.no_position_strategy import *
from strategy.pick_stock.price_volume_strategy import *
from strategy.pick_stock.revert_strategy import *
from strategy.pick_stock.turnover_rate_strategy import *
from strategy.pick_stock.average_strategy import *


def get_strategy_function(strategy_name):

    # 振幅类
    if strategy_name == "20日平均振幅策略":
        pick_stock_strategy = amplitude_20_day_strategy
    elif strategy_name == "单日振幅20日均值策略":
        pick_stock_strategy = one_day_amplitude_20_day_average_strategy

    # 市值类
    elif strategy_name == "小市值策略":
        pick_stock_strategy = small_cap_strategy
    elif strategy_name == "大市值策略":
        pick_stock_strategy = large_cap_strategy
    elif strategy_name == "低价选股策略":
        pick_stock_strategy = low_price_strategy
    elif strategy_name == "低价选股策略_百分比":
        pick_stock_strategy = low_price_pct_strategy
    elif strategy_name == "垃圾股策略":
        pick_stock_strategy = low_price_small_cap_strategy
    elif strategy_name == "小市值策略_量价优化1":
        pick_stock_strategy = small_cap_strategy_pv_opt_1

    # 惯性类
    elif strategy_name == "惯性策略":
        pick_stock_strategy = consistent_strategy

    # 反转类
    elif strategy_name == "反转策略":
        pick_stock_strategy = revert_strategy

    # 量价类
    elif strategy_name == "量价相关性策略":
        pick_stock_strategy = price_volume_strategy
    elif strategy_name == "量比策略":
        pick_stock_strategy = volume_ratio_strategy
    elif strategy_name == "香农短线量价策略":
        pick_stock_strategy = wr_bias_strategy
    elif strategy_name == "挖掘放量待涨个股策略":
        pick_stock_strategy = volume_turnover_rate_strategy

    # 换手率策略
    elif strategy_name == "换手率策略":
        pick_stock_strategy = turnover_rate_strategy

    # 均线策略
    elif strategy_name == "单均线策略20日":
        pick_stock_strategy = average_20_day_strategy
    elif strategy_name == "单均线策略5日":
        pick_stock_strategy = average_5_day_strategy

    # 空仓策略
    elif strategy_name == "空仓策略":
        pick_stock_strategy = no_position_strategy

    else:
        raise Exception("尚无此策略！")

    return pick_stock_strategy
