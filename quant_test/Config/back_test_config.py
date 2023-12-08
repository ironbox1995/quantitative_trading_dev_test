# -*- coding: utf-8 -*-
# ====================回测配置====================

from backtest.utils import get_current_date

# ==========策略配置==========
strategy_li = ["小市值策略"]
# strategy_li = ["空仓策略"]
period_type_li = ['W']
select_stock_num_li = [3]
date_start = '2010-01-01'
date_end = get_current_date()


# ==========止盈止损配置==========
# 涨跌停止盈止损
limit_up_take_profit = False
limit_down_stop_loss = False


# ==========择时配置==========
# pick_time_li = ["双均线择时", "MICD择时", "SROC择时", "ENV择时", "MTM择时", "DPO择时", "T3择时", "BBI择时", "PMO择时",
#                 "PO择时", "RSIH择时", "WMA择时", "TMA择时", "MACD择时", "KDJ择时", "ARRON择时", "无择时"]
# pick_time_li = ["WMA择时", "无择时"]
# 无创业无科创前提下：
pick_time_mtd_dct = {"小市值策略": "DPO择时",  # DPO择时
                     "小市值策略_分箱优化1": "无择时",  # WMA择时
                     "小市值策略_分箱优化4": "无择时",  # DPO择时
                     "小市值策略_量价优化1": "无择时",  # MTM择时
                     "小市值策略_分箱优化5": "SROC择时",
                     "香农短线量价策略": "DPO择时",
                     "低价小市值策略": "MTM择时",
                     "多因子量价策略2": "DPO择时",
                     "低回撤单因子组合策略": "DPO择时",
                     "均线偏离与流通市值策略": "MTM择时",
                     "换手率筛选多因子排序策略": "MTM择时",
                     "非高价股选股策略": "双均线择时",
                     "大市值策略": "无择时",
                     "空仓策略": "无择时",
                     "Q学习并联策略": "无择时"
                     }

# ==========Q学习配置==========
sub_strategy_li = ["小市值策略", "小市值策略_量价优化1", "小市值策略_分箱优化1", "小市值策略_分箱优化4"]
ALPHA = 0.9
eps = 0.1
