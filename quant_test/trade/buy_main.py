# -*- coding: utf-8 -*-
from trade.place_order_main import *
from trade.script.script_utils import *
from utils_global.dingding_message import *

# 周一早上执行这个
if __name__ == "__main__":
    try:
        if first_workday_in_period():  # 判断是周一且不止有一个交易日
            run_strategy_buy()
            send_dingding("交易播报：买入成功！具体交易细节见日志。")
        else:
            send_dingding("交易播报：今天不符合买入日条件。")
    except Exception as e:
        print(e)
        send_dingding("交易播报：买入失败！请详细查看后台了解具体原因！")
