# -*- coding: utf-8 -*-
from trade.place_order_main import *
from trade.script.script_utils import *
from utils_global.dingding_message import *

# 周五下午执行这个
if __name__ == "__main__":
    try:
        if last_workday_in_period() or force_run:
            send_dingding("交易播报：开始执行卖出委托！")
            cash_amount_before, cash_amount_after = run_strategy_sell()
            save_info_dct = {"日期": datetime.today().date(), "交易前现金金额": cash_amount_before,
                             "交易后现金金额": cash_amount_after, "备注": "卖出"}
            save_to_csv(save_info_dct)
            send_dingding("交易播报：卖出委托成功！具体交易细节见日志。")
        else:
            send_dingding("交易播报：今天不符合卖出日条件。")
    except Exception as e:
        print(e)
        send_dingding("交易播报：卖出失败！请详细查看后台了解具体原因！")
