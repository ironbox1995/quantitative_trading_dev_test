# -*- coding: utf-8 -*-

from utils import *

strategy_li = ["小市值策略", "量价相关性策略", "小市值策略_量价优化1"]
# strategy_li = ["小市值策略_量价优化1"]
period_type_li = ['W']
select_stock_num_li = [3]
date_start = '2010-01-01'
# date_end = '2023-07-07'
date_end = get_current_date()
pick_time_mtd = "双均线择时"