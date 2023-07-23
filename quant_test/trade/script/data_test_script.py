# -*- coding: utf-8 -*-
from trade.script.dingding_message import *
from trade.script.script_utils import *


def execute_multiple_files(file_paths):
    exe_li = ["数据获取", "数据处理", "回测", "最新结果输出"]
    for i, file_path in enumerate(file_paths):
        print("执行：{}".format(exe_li[i]))
        try:
            execute_script_in_virtualenv(file_path)
        except:
            send_dingding("交易播报：执行{}失败，后续流程已阻断，请查看后台了解详细原因。".format(exe_li[i]))
            break
        else:
            send_dingding("交易播报：执行{}成功！".format(exe_li[i]))


# 周六执行这个
if __name__ == "__main__":
    file_paths = ['F:\\quantitative_trading\\quant_formal\\data\\download_data_main.py'
        , 'F:\\quantitative_trading\\quant_formal\\data\\data_processing_main.py'
        , 'F:\\quantitative_trading\\quant_formal\\backtest\\backtest_pick_stock.py'
        , 'F:\\quantitative_trading\\quant_formal\\backtest\\latest_result.py']
    execute_multiple_files(file_paths)
