# -*- coding: utf-8 -*-
from trade.place_order_main import *
from trade.script.script_utils import *
from utils_global.dingding_message import *
from trade.buy_repo_main import *

# 周一早上执行这个
if __name__ == "__main__":
    try:
        if first_workday_in_period():  # 判断是周一且不止有一个交易日
            send_dingding("交易播报：开始执行买入股票委托！")
            cash_amount = run_strategy_buy()
            save_info_dct = {"日期": datetime.datetime.today().date(), "现金金额": cash_amount, "备注": "本周买入前金额"}
            save_to_csv(save_info_dct)
            send_dingding("交易播报：买入股票委托成功！")

            # 延迟到九点半
            while datetime.datetime.now().strftime("%H:%M:%S") < "09:35:01":  # 一开始利率不稳定
                print(datetime.datetime.now().strftime("%H:%M:%S"), end="\r")
            send_dingding("交易播报：开始执行买入逆回购委托！")
            buy_reverse_repo(code='131809.SZ')  # 买入四天期逆回购
            send_dingding("交易播报：买入逆回购委托成功！")

        else:
            send_dingding("交易播报：今天不符合买入日条件。")
    except Exception as e:
        print(e)
        send_dingding("交易播报：买入失败！请详细查看后台了解具体原因！")
