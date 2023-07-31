# -*- coding: utf-8 -*-
"""
《邢不行-2021新版|Python股票量化投资课程》
author: 邢不行
微信: xbx9585

QMT自动交易案例
需要打开极简版QMT
"""
import time
import datetime
from xtquant import xtconstant  # qmt常量
from xtquant.xttype import StockAccount  # 证券账户
from xtquant.xttrader import XtQuantTrader  # 交易接口
from xtquant import xtdata
from trade.load_strategy import *
from trade.query import *


"""
交易案例
指定一批股票，均仓买入。

如何定时运行本脚本？
https://mp.weixin.qq.com/s/vrv-PniBGxEerJ44AV0jcw
"""


def run_strategy_buy():

    # ========== 初始化交易接口 ==========
    path = 'F:\\中航证券QMT实盘-交易端\\userdata_mini'  # 极简版QMT的路径
    session_id = 100001  # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号（自己随便写）
    xt_trader = XtQuantTrader(path, session_id)  # 创建API实例
    user = StockAccount('010400007212', 'STOCK')  # 创建股票账户
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    print('链接成功' if connect_result == 0 else '链接失败')
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(user)
    print('订阅成功' if subscribe_result == 0 else '订阅失败')
    account_res = xt_trader.query_stock_asset(user)

    print("连接时间：{}".format(datetime.datetime.now()))

    cash_amount = account_res.cash
    print("周一开单前现金量：{}".format(cash_amount))
    # ========== 策略配置 ==========
    # buy_stock_list = ['002708.SZ', '003023.SZ', '002094.SZ']
    # buy_amount = 100000  # 0表示使用所有可用资金买入
    buy_stock_list, buy_amount = load_strategy_result(cash_amount=account_res.cash)

    if len(buy_stock_list) > 0:
        # ========== 策略运行 ==========
        # 批量订阅数据
        for stock in buy_stock_list:
            sub_id = xtdata.subscribe_quote(stock, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
            print(f'{stock}订阅成功，订阅号：{sub_id}')
        time.sleep(3)

        # 计算买入股票的下单金额
        # account_res = xt_trader.query_stock_asset(user)
        if buy_amount == 0:  # 全仓买入
            single_stock_amount = account_res.cash / len(buy_stock_list)
        else:  # 指定金额买入
            single_stock_amount = min(buy_amount, account_res.cash) / len(buy_stock_list)

        print('正在执行买入操作')
        for buy in buy_stock_list:
            # 获取最新价格
            last_price = xtdata.get_full_tick([buy])[buy]['lastPrice']
            # 计算下单量：普通板块，一手100股，最低1手。科创板最低200股，超过200以后最低1股。（科创板处理流程当做作业自行实现）
            volume = single_stock_amount / last_price
            volume = int(volume - volume % 100)
            if volume < 100:
                print(f'{buy}下单量不足100股')
                continue
            # 按照开盘价下单（实际这样可能会存在无法成交的情况）
            last_price = xtdata.get_full_tick([buy])[buy]['lastPrice']
            order_id = xt_trader.order_stock(user, buy, xtconstant.STOCK_BUY, volume, xtconstant.FIX_PRICE,
                                             last_price, 'strategy', 'remark')
            if order_id != -1:
                print(f'{buy}下单成功，下单价格：{last_price}，下单量：{volume}')
                print("下单时间：{}".format(datetime.datetime.now()))
            else:
                print(f'{buy}下单失败！')
                print("下单时间：{}".format(datetime.datetime.now()))
    else:
        print("本周期无下单目标。")
    return cash_amount


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
    print('链接成功' if connect_result == 0 else '链接失败')
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(user)
    print('订阅成功' if subscribe_result == 0 else '订阅失败')

    print("连接时间：{}".format(datetime.datetime.now()))

    sell_stock_dct = query_stock_with_position(xt_trader, user)

    for sell_stock, sell_amount in sell_stock_dct.items():

        # 订阅数据
        sub_id = xtdata.subscribe_quote(sell_stock, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
        print(f'{sell_stock}订阅成功，订阅号：{sub_id}')
        time.sleep(3)

        print('正在执行卖出操作')

        # 按照收盘价卖出（实际这样可能会存在无法成交的情况）
        last_price = xtdata.get_full_tick([sell_stock])[sell_stock]['lastPrice']
        order_id = xt_trader.order_stock(user, sell_stock, xtconstant.STOCK_SELL, sell_amount, xtconstant.FIX_PRICE,
                                         last_price, 'strategy', 'remark')
        if order_id != -1:
            print(f'{sell_stock}卖出成功，卖出价格：{last_price}，卖出量：{sell_amount}')
            print("卖出时间：{}".format(datetime.datetime.now()))
        else:
            print(f'{sell_stock}卖出失败！')
            print("卖出时间：{}".format(datetime.datetime.now()))

    account_res = xt_trader.query_stock_asset(user)
    print("周五平仓后现金量：{}".format(account_res.cash))

    return account_res.cash


if __name__ == "__main__":

    # 周一早上执行这个
    run_strategy_buy()

    # 周五下午执行这个
    run_strategy_sell()