# -*- coding: utf-8 -*-

from utils import *

strategy_li = ["小市值策略", "香农短线量价策略", "低价小市值策略", "多因子量价策略2", "低回撤单因子组合策略",
               "均线偏离与流通市值策略", "小市值策略_量价优化1", "换手率筛选多因子排序策略", "非高价股选股策略"]
# strategy_li = ["小市值策略"]
period_type_li = ['W']
select_stock_num_li = [3]
date_start = '2010-01-01'
date_end = get_current_date()

# 无创业无科创前提下：
pick_time_mtd_dct = {"小市值策略": "MTM择时",
                     "香农短线量价策略": "DPO择时",
                     "低价小市值策略": "MTM择时",
                     "多因子量价策略2": "DPO择时",
                     "低回撤单因子组合策略": "DPO择时",
                     "均线偏离与流通市值策略": "MTM择时",
                     "小市值策略_量价优化1": "MTM择时",
                     "换手率筛选多因子排序策略": "MTM择时",
                     "非高价股选股策略": "双均线择时"
                     }
ALPHA = 0.9
eps = -1
