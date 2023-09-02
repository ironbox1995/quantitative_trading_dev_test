# -*- coding: utf-8 -*-
from trade.script.script_utils import *
from utils_global.global_config import *


def execute_multiple_files(file_paths):
    exe_li = ["数据获取", "数据处理", "回测", "最新结果输出"]
    for i, file_path in enumerate(file_paths):
        print("开始执行：{}".format(exe_li[i]))
        execute_script_in_virtualenv(file_path)


# 周六执行这个
if __name__ == "__main__":
    file_paths = [r'{}\data\download_data_main.py'.format(project_path),
                  r'{}\data\data_processing_main.py'.format(project_path),
                  r'{}\backtest\backtest_pick_stock.py'.format(project_path),
                  r'{}\backtest\latest_result.py'.format(project_path)]
    execute_multiple_files(file_paths)
