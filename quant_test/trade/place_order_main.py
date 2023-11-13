# -*- coding: utf-8 -*-
"""
《邢不行-2021新版|Python股票量化投资课程》
author: 邢不行
微信: xbx9585

QMT自动交易案例
需要打开极简版QMT
"""
from xtquant import xtconstant  # qmt常量
from xtquant.xttype import StockAccount  # 证券账户
from xtquant.xttrader import XtQuantTrader  # 交易接口
from decimal import Decimal, ROUND_HALF_UP
from xtquant import xtdata
from trade.query import *
from trade.logger.logger import record_log
from Config.trade_config import repo_code, buy_reverse_repo
from Config.global_config import total_position
import datetime

"""
如何定时运行本脚本？
https://mp.weixin.qq.com/s/vrv-PniBGxEerJ44AV0jcw
"""


def calculate_order_quantity(code, order_volume):
    """
    处理下单股数
    计算下单量：普通板块，一手100股，最低1手。科创板最低200股，超过200以后最低1股。
    :param code:
    :param order_volume:
    :return:
    """
    # 对科创板进行特殊处理
    if code[:2] == '68' and code[-2:] == 'SH':
        order_volume = int(order_volume)
        if order_volume < 200:
            return 0
    else:
        order_volume = int(order_volume / 100) * 100
        if order_volume < 100:
            return 0
    return order_volume


def cal_limit_up(code, last_price):
    # 对科创板进行特殊处理
    if code[:2] == '68' and code[-2:] == 'SH':
        limit_up = last_price * 1.2
    elif code[:2] == '30' and code[-2:] == 'SZ':
        limit_up = last_price * 1.2
    else:
        limit_up = last_price * 1.1

    return float(Decimal(limit_up * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100)


def place_stock_order(xt_trader, user, cash_amount, all_buy_stock, use_limit_up):
    for strategy_tup in all_buy_stock:
        buy_stock_list = strategy_tup[0]
        strategy_buy_amount = strategy_tup[1] * cash_amount
        strategy_name = strategy_tup[2]
        record_log(f'策略{strategy_name}下单开始！使用{"涨停价" if use_limit_up else "昨收价"}下单')
        if len(buy_stock_list) > 0:
            # 批量订阅数据
            for stock in buy_stock_list:
                sub_id = xtdata.subscribe_quote(stock, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
                record_log(f'{stock}订阅成功，订阅号：{sub_id}')
            # 计算买入股票的下单金额
            single_stock_amount = strategy_buy_amount / len(buy_stock_list)
            record_log('正在执行买入操作')
            for buy_stock in buy_stock_list:
                # 计算最新价和涨停价，并判断使用哪个下单
                last_price = xtdata.get_full_tick([buy_stock])[buy_stock]['lastPrice']  # 获取最新价格
                limit_up = cal_limit_up(buy_stock, last_price)  # 计算涨停价
                if use_limit_up:
                    order_price = limit_up
                else:
                    order_price = last_price
                volume = calculate_order_quantity(buy_stock, single_stock_amount / order_price)
                if volume < 100:
                    record_log(f'{buy_stock}下单量不足')
                    continue
                for _ in range(5):
                    try:
                        # 按照之前计算的价格下单
                        order_id = xt_trader.order_stock(user, buy_stock, xtconstant.STOCK_BUY, volume, xtconstant.FIX_PRICE,
                                                         order_price, 'weekly strategy', 'remark')
                        if order_id != -1:
                            record_log(f'开仓{buy_stock}委托成功，报价类型：{xtconstant.FIX_PRICE}，下单价格：{order_price}，下单量：{volume}')
                            record_log("下单时间：{}".format(datetime.datetime.now()))
                            break
                        else:
                            record_log(f'开仓{buy_stock}下单失败！')
                            raise Exception(f'开仓{buy_stock}下单失败！')
                    except Exception as e:
                        record_log(f'开仓{buy_stock}下单失败！')
                        print(e)
                        pass
            record_log(f"策略:{strategy_name}已完成全部下单。")
        else:
            record_log(f"本周期策略:{strategy_name}不进行下单。")


def place_repo_order(xt_trader, user):
    if buy_reverse_repo:
        # 开始进行逆回购下单
        account_res = xt_trader.query_stock_asset(user)  # 重新计算剩余现金
        cash_amount_repo = account_res.cash
        # 计算下单量：
        buy_volume = (cash_amount_repo // 1000) * 1000
        if buy_volume < 1000:
            record_log(f'{repo_code}下单量不足，程序退出')
        else:
            record_log("买入逆回购现金量：{}".format(buy_volume))
            sub_id = xtdata.subscribe_quote(repo_code, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
            record_log(f'{repo_code}订阅成功，订阅号：{sub_id}')

            # 按照最新价下单
            for i in range(5):
                try:
                    order_id = xt_trader.order_stock(user, repo_code, xtconstant.CREDIT_SLO_SELL, int(buy_volume), xtconstant.LATEST_PRICE, 0,
                                                    'weekly strategy', 'remark')
                    if order_id != -1:
                        record_log(f'逆回购{repo_code}下单成功，下单量：{buy_volume}')
                        record_log("下单时间：{}".format(datetime.datetime.now()))
                        break
                    else:
                        record_log(f'{repo_code}下单失败！')
                        raise Exception(f'{repo_code}下单失败！')
                except Exception as e:
                    record_log(f'第{i}次尝试下单{repo_code}失败！')
                    print(e)


def run_strategy_buy(all_buy_stock):
    # ========== 初始化交易接口 ==========
    path = 'F:\\中航证券QMT实盘-交易端\\userdata_mini'  # 极简版QMT的路径
    session_id = 100001  # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号（自己随便写）
    xt_trader = XtQuantTrader(path, session_id)  # 创建API实例
    user = StockAccount('010400007212', 'STOCK')  # 创建股票账户
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(user)
    if subscribe_result != 0 or connect_result != 0:
        record_log('交易播报：链接或订阅失败，程序已退出', send=True)
        exit()
    else:
        record_log('链接并订阅成功')
    record_log("连接时间：{}".format(datetime.datetime.now()))

    # ========== 可用现金量计算 ==========
    account_res = xt_trader.query_stock_asset(user)
    cash_amount_before = account_res.cash
    record_log("周一开单前现金量：{}".format(cash_amount_before))
    if total_position >= 0:
        cash_amount = min(cash_amount_before, total_position)
    else:
        cash_amount = cash_amount_before

    # ========== 开始买入股票 ==========
    # 为确保成交率，使用涨停价买入
    place_stock_order(xt_trader, user, cash_amount, all_buy_stock, use_limit_up=True)

    # 为确保资金使用率，使用昨收价买入
    # 重新计算可用现金量
    account_res = xt_trader.query_stock_asset(user)
    cash_amount_to_use = account_res.cash
    if total_position >= 0:
        cash_amount = min(cash_amount_to_use, total_position)
    else:
        cash_amount = cash_amount_to_use
    place_stock_order(xt_trader, user, cash_amount, all_buy_stock, use_limit_up=False)

    # ========== 开始买入逆回购 ==========
    place_repo_order(xt_trader, user)

    # ========== 计算剩余现金 ==========
    account_res = xt_trader.query_stock_asset(user)
    cash_amount_after = account_res.cash
    record_log("周一开单后现金量：{}".format(cash_amount_after))

    return cash_amount_before, cash_amount_after


def run_strategy_sell():

    # ========== 初始化交易接口 ==========
    path = 'F:\\中航证券QMT实盘-交易端\\userdata_mini'  # 极简版QMT的路径
    session_id = 100002  # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号（自己随便写）
    xt_trader = XtQuantTrader(path, session_id)  # 创建API实例
    user = StockAccount('010400007212', 'STOCK')  # 创建股票账户
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(user)
    if subscribe_result != 0 or connect_result != 0:
        record_log('交易播报：链接或订阅失败，程序已退出', send=True)
        exit()
    else:
        record_log('链接并订阅成功')
    record_log("连接时间：{}".format(datetime.datetime.now()))

    account_res = xt_trader.query_stock_asset(user)
    cash_amount_before = account_res.cash
    record_log("周五平仓前现金量：{}".format(cash_amount_before))
    sell_stock_dct = query_stock_with_position(xt_trader, user)

    # ========== 开始卖出股票 ==========
    for sell_stock, sell_amount in sell_stock_dct.items():

        # 订阅数据
        sub_id = xtdata.subscribe_quote(sell_stock, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
        record_log(f'{sell_stock}订阅成功，订阅号：{sub_id}')
        record_log('正在执行卖出操作')
        # 按照最新盘价卖出
        for _ in range(5):
            last_price = xtdata.get_full_tick([sell_stock])[sell_stock]['lastPrice']
            try:
                order_id = xt_trader.order_stock(user, sell_stock, xtconstant.STOCK_SELL, sell_amount, xtconstant.LATEST_PRICE,
                                                 0, 'weekly strategy', 'remark')
                if order_id != -1:
                    record_log(f'平仓{sell_stock}委托成功，报价类型：{xtconstant.FIX_PRICE}，委托价格约为：{last_price}，委托量：{sell_amount}')
                    break
                else:
                    record_log(f'平仓{sell_stock}委托失败！')
                    raise Exception(f'平仓{sell_stock}委托失败！')
            except:
                pass

    # 推到当日交易时间结束之后再查询余额
    while datetime.datetime.now().strftime("%H:%M:%S") < "15:00:30":
        print(datetime.datetime.now().strftime("%H:%M:%S"), end="\r")

    account_res = xt_trader.query_stock_asset(user)
    cash_amount_after = account_res.cash
    record_log("周五平仓后现金量：{}".format(cash_amount_after))

    return cash_amount_before, cash_amount_after


# if __name__ == "__main__":
#     all_buy_stock = load_strategy_result()
#
#     # 周一早上执行这个
#     run_strategy_buy(all_buy_stock)
#
#     # 周五下午执行这个
#     run_strategy_sell()
