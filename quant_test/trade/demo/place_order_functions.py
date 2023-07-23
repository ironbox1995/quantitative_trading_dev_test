import time
from xtquant import xtconstant  # qmt常量
from xtquant.xttype import StockAccount  # 证券账户
from xtquant.xttrader import XtQuantTrader  # 交易接口
from xtquant import xtdata


def initialize_xt_trader(session_id):
    """
    初始化交易接口
    :param session_id: session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号（自己随便写，如：123456）
    :return:
    """
    path = r'F:\中航证券QMT实盘-交易端\userdata_mini'  # 极简版QMT的路径
    xt_trader = XtQuantTrader(path, session_id)  # 创建API实例
    user = StockAccount('', 'STOCK')  # 创建股票账户 TODO: 看下这个参数配置

    # 启动交易线程
    xt_trader.start()

    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    print('链接成功' if connect_result == 0 else '链接失败')

    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(user)
    print('订阅成功' if subscribe_result == 0 else '订阅失败')

    return xt_trader, user, connect_result, subscribe_result,


def buy_order(xt_trader, user, buy_stock, buy_amount, price_type=xtconstant.FIX_PRICE):
    """
    对单个股票下单笔买入订单。
    最新价：LATEST_PRICE
    指定价/限价：FIX_PRICE
    上海最优五档即时成交剩余撤销：MARKET_SH_CONVERT_5_CANCEL
    上海最优五档即时成交剩余转限价：MARKET_SH_CONVERT_5_LIMIT
    深圳对手方最优价格：MARKET_PEER_PRICE_FIRST
    深圳本方最优价格：MARKET_MINE_PRICE_FIRST
    深圳即时成交剩余撤销：MARKET_SZ_INSTBUSI_RESTCANCEL
    深圳最优五档即时成交剩余撤销：MARKET_SZ_CONVERT_5_CANCEL
    深圳全额成交或撤销：MARKET_SZ_FULL_OR_CANCEL
    """

    # ========== 策略运行 ==========
    # 订阅数据
    sub_id = xtdata.subscribe_quote(buy_stock, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
    print(f'{buy_stock}订阅成功，订阅号：{sub_id}')
    time.sleep(3)

    # 计算买入股票的下单金额
    account_res = xt_trader.query_stock_asset(user)
    if buy_amount == 0:  # 全仓买入
        single_stock_amount = account_res.cash
    else:  # 指定金额买入
        single_stock_amount = min(buy_amount, account_res.cash)

    print('正在执行买入操作')
    # 获取最新价格
    last_price = xtdata.get_full_tick([buy_stock])[buy_stock]['lastPrice']
    # 计算下单量：普通板块，一手100股，最低1手。科创板最低200股，超过200以后最低1股。（科创板处理流程当做作业自行实现）
    volume = single_stock_amount / last_price
    volume = int(volume - volume % 100)
    if volume < 100:
        print(f'{buy_stock}下单量不足100股')
        order_id = -1
    else:
        # 按照开盘价下单（实际这样可能会存在无法成交的情况） TODO：怎么办呢？
        last_price = xtdata.get_full_tick([buy_stock])[buy_stock]['lastPrice']
        order_id = xt_trader.order_stock(user, buy_stock, xtconstant.STOCK_BUY, volume, price_type,
                                         last_price, 'strategy', 'remark')
        if order_id != -1:
            print(f'{buy_stock}下单成功，下单价格：{last_price}，下单量：{volume}')

    return order_id


def cancel_order(xt_trader, user, order_id):
    # 撤单，撤回指定的订单。不能一下子全撤
    cancel_res = xt_trader.cancel_order_stock(user, order_id)
    print(f'撤单成功' if cancel_res != -1 else '撤单失败')
    return cancel_res


# TODO：检查函数是否正确
def sell_order(xt_trader, user, sell_stock, sell_amount, price_type=xtconstant.FIX_PRICE):
    """
    对单个股票进行单笔卖出。
    """
    # 订阅数据 TODO:买卖之前一定要订阅吗？
    sub_id = xtdata.subscribe_quote(sell_stock, period='tick', count=-1)  # 1个tick是3s，5分钟是100个tick
    print(f'{sell_stock}订阅成功，订阅号：{sub_id}')
    time.sleep(3)

    print('正在执行卖出操作')

    # 按照收盘价卖出（实际这样可能会存在无法成交的情况） TODO：怎么办呢？
    last_price = xtdata.get_full_tick([sell_stock])[sell_stock]['lastPrice']
    order_id = xt_trader.order_stock(user, sell_stock, xtconstant.STOCK_SELL, sell_amount, price_type,
                                     last_price, 'strategy', 'remark')
    if order_id != -1:
        print(f'{sell_stock}卖出成功，卖出价格：{last_price}，卖出量：{sell_amount}')

    return order_id


