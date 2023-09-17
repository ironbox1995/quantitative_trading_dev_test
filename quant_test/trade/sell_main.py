# -*- coding: utf-8 -*-
from trade.place_order_main import *
from trade.script.script_utils import *
from utils_global.dingding_message import *

# 周五下午执行这个
if __name__ == "__main__":
    try:
        if last_workday_in_period():
            send_dingding("交易播报：开始执行卖出委托！")
            cash_amount = run_strategy_sell()
            save_info_dct = {"日期": datetime.today().date(), "现金金额": cash_amount, "备注": "本周卖出后金额"}
            save_to_csv(save_info_dct)
            send_dingding("交易播报：卖出委托成功！具体交易细节见日志。")
        else:
            send_dingding("交易播报：今天不符合卖出日条件。")
    except Exception as e:
        print(e)
        send_dingding("交易播报：卖出失败！请详细查看后台了解具体原因！")
