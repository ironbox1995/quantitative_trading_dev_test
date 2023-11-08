# 导入选股策略
from strategy.pick_stock.capitalization_strategy import *
from strategy.pick_stock.no_position_strategy import *

# 导入择时策略
from strategy.pick_time.MA_signal import *
from strategy.pick_time.index_signal import *


def get_pick_stock_strategy(strategy_name):
    """
    获取选股策略
    """
    # 空仓策略
    if strategy_name == "空仓策略":
        pick_stock_strategy = no_position_strategy

    # 市值类
    elif strategy_name == "小市值策略":
        pick_stock_strategy = small_cap_strategy
    elif strategy_name == "小市值策略_分箱优化1":
        pick_stock_strategy = small_cap_bin_optimized1
    elif strategy_name == "小市值策略_量价优化1":
        pick_stock_strategy = small_cap_strategy_pv_opt_1

    else:
        raise Exception("尚无此策略或经验证不可用！")

    return pick_stock_strategy


def get_pick_time_strategy(select_stock, pick_time_mtd):
    """
    获取择时策略
    """
    if pick_time_mtd == "双均线择时":
        select_stock, latest_signal = MA_signal(select_stock, para=(1, 3))
    elif pick_time_mtd == "MTM择时":
        select_stock, latest_signal = MTM_signal(select_stock, 2)
    elif pick_time_mtd == "DPO择时":
        select_stock, latest_signal = DPO_signal(select_stock, 2)
    elif pick_time_mtd == "WMA择时":
        select_stock, latest_signal = WMA_signal(select_stock, 5)
    else:
        raise Exception("暂无此择时方法！")
    return select_stock, latest_signal
