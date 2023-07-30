# -*- coding: utf-8 -*-

from utils import *

strategy_li = ["小市值策略", "小市值策略_量价优化1", "香农短线量价策略", "低价小市值策略", "多因子量价策略2"]
# strategy_li = ["多因子量价策略1"]
period_type_li = ['W']
select_stock_num_li = [3]
# date_start = '2020-01-01'
date_start = '2010-01-01'
# date_end = '2023-07-07'
date_end = get_current_date()
pick_time_mtd = "双均线择时"
ALPHA = 0.7
