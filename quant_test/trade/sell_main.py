# -*- coding: utf-8 -*-
from trade.place_order_main import *
from trade.script.script_utils import *
from utils_global.dingding_message import *

# 周五下午执行这个
if __name__ == "__main__":
    try:
        if last_workday_in_period():
            run_strategy_sell()
            send_dingding("交易播报：卖出成功！具体交易细节见日志。")
        else:
            send_dingding("交易播报：今天不符合卖出日条件。")
    except Exception as e:
        print(e)
        send_dingding("交易播报：卖出失败！请详细查看后台了解具体原因！")
