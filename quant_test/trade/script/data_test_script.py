# -*- coding: utf-8 -*-
from trade.script.script_utils import *


def execute_multiple_files(file_paths):
    exe_li = ["数据获取", "数据处理", "回测", "最新结果输出"]
    for i, file_path in enumerate(file_paths):
        print("开始执行：{}".format(exe_li[i]))
        execute_script_in_virtualenv(file_path)


# 周六执行这个
if __name__ == "__main__":
    file_paths = ['F:\\quantitative_trading_dev_test\\quant_test\\data\\download_data_main.py'
        , 'F:\\quantitative_trading_dev_test\\quant_test\\data\\data_processing_main.py'
        , 'F:\\quantitative_trading_dev_test\\quant_test\\backtest\\backtest_pick_stock.py'
        , 'F:\\quantitative_trading_dev_test\\quant_test\\backtest\\combination_of_strategies.py']
    execute_multiple_files(file_paths)
