# -*- coding: utf-8 -*-
from trade.place_order_main import *

# 周五下午执行这个
if __name__ == "__main__":
    file_path = "实盘日志.txt"
    run_strategy_sell(file_path)
