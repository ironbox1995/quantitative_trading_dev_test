# -*- coding: utf-8 -*-
# ==========策略配置==========
strategy_part_dct = {"小市值策略": 0.5, "小市值策略_分箱优化1": 0.5}  # 策略及资金权重
strategy_li = strategy_part_dct.keys()
period_type = 'W'  # 数据周期
select_stock_num = 3  # 选股数量

# ==========择时配置==========
pick_time_switch = True  # 择时开关
# 无创业无科创前提下：
pick_time_mtd_dct = {"小市值策略": "无择时",  # DPO择时
                     "小市值策略_分箱优化1": "WMA择时",
                     "小市值策略_量价优化1": "MTM择时",
                     "香农短线量价策略": "DPO择时",
                     "低价小市值策略": "MTM择时",
                     "多因子量价策略2": "DPO择时",
                     "低回撤单因子组合策略": "DPO择时",
                     "均线偏离与流通市值策略": "MTM择时",
                     "换手率筛选多因子排序策略": "MTM择时",
                     "非高价股选股策略": "双均线择时"
                     }

# ==========逆回购配置==========
"""
每天买入一天期逆回购：https://bbs.quantclass.cn/thread/3401
'131810.SZ' --一天期
'131809.SZ' --四天期
"""
buy_reverse_repo = True
repo_code = '131809.SZ'

# ==========止损配置==========
# strategy_stop_loss_point_dct = {"小市值策略": 1, "小市值策略_分箱优化1": 1}
strategy_stop_loss_point_dct = {"{}_{}_选{}_{}".format("小市值策略", period_type, select_stock_num, "无择时"): 1,
                                "{}_{}_选{}_{}".format("小市值策略_分箱优化1", period_type, select_stock_num, "WMA择时"): 1}
draw_down_warning_point = 0.8

# ==========Q学习配置==========
Q_strategy_li = ["小市值策略", "香农短线量价策略", "低价小市值策略", "多因子量价策略2", "低回撤单因子组合策略",
                 "均线偏离与流通市值策略", "小市值策略_量价优化1", "换手率筛选多因子排序策略", "非高价股选股策略"]

eps = -1
