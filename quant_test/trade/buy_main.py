# -*- coding: utf-8 -*-
from trade.place_order_main import *
from trade.script.script_utils import *
from trade.load_strategy import *


# 周一早上执行这个
if __name__ == "__main__":
    try:
        if first_workday_in_period() or force_run:  # 判断是周一且不止有一个交易日
            send_dingding("交易播报：开始执行买入股票委托！")
            all_buy_stock = load_strategy_result()
            cash_amount_before, cash_amount_after = run_strategy_buy(all_buy_stock)
            save_info_dct = {"日期": datetime.today().date(), "交易前现金金额": cash_amount_before,
                             "交易后现金金额": cash_amount_after, "备注": "买入"}
            save_to_csv(save_info_dct)
            send_dingding("交易播报：买入股票委托成功！")

            # 延迟到九点半
            # while datetime.datetime.now().strftime("%H:%M:%S") < "09:35:01":  # 一开始利率不稳定
            #     print(datetime.datetime.now().strftime("%H:%M:%S"), end="\r")
            # send_dingding("交易播报：开始执行买入逆回购委托！")
            # buy_reverse_repo(code='131809.SZ')  # 买入四天期逆回购
            # send_dingding("交易播报：买入逆回购委托成功！")

        else:
            send_dingding("交易播报：今天不符合买入日条件。")
    except Exception as e:
        print(e)
        send_dingding("交易播报：买入失败！请详细查看后台了解具体原因！")
