from xtquant import xtdata
from xtquant import xtconstant  # qmt常量
from xtquant.xttype import StockAccount  # 证券账户
from xtquant.xttrader import XtQuantTrader  # 交易接口
from trade.logger.logger import record_log
import datetime


def buy_reverse_repo(remain_money, code='131810.SZ'):
    """
    每天买入一天期逆回购：https://bbs.quantclass.cn/thread/3401
    TODO：后续可以考虑反弹下单之类的手段
    '131810.SZ'
    :return:
    """
    # ========== 初始化交易接口 ==========
    path = 'F:\\中航证券QMT实盘-交易端\\userdata_mini'  # 极简版QMT的路径
    session_id = 100001  # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号（自己随便写）
    xt_trader = XtQuantTrader(path, session_id)  # 创建API实例
    user = StockAccount('010400007212', 'STOCK')  # 创建股票账户
    # 启动交易线程
    xt_trader.start()
    # 建立交易连接，返回0表示连接成功
    connect_result = xt_trader.connect()
    record_log('链接成功' if connect_result == 0 else '链接失败')
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    subscribe_result = xt_trader.subscribe(user)
    record_log('订阅成功' if subscribe_result == 0 else '订阅失败')
    account_res = xt_trader.query_stock_asset(user)

    record_log("连接时间：{}".format(datetime.datetime.now()))

    cash_amount = account_res.cash
    record_log("买入逆回购现金量：{}".format(cash_amount))

    # 获取最新价格
    last_price = xtdata.get_full_tick([code])[code]['lastPrice']
    # 计算下单量：普通板块，一手100股，最低1手。科创板最低200股，超过200以后最低1股。
    buy_volume = (remain_money//1000) * 1000
    if buy_volume < 1000:
        record_log(f'{code}下单量不足')

    # 按照最新价下单
    for _ in range(5):
        try:
            # order_id = xt_trader.order_stock(user, buy, xtconstant.STOCK_BUY, volume, xtconstant.LATEST_PRICE,
            #                                  0, 'weekly strategy', 'remark')
            last_price = xtdata.get_full_tick([code])[code]['lastPrice']
            order_id = xt_trader.order_stock(user, code, xtconstant.STOCK_BUY, buy_volume, xtconstant.FIX_PRICE,
                                             last_price, 'weekly strategy', 'remark')
            if order_id != -1:
                record_log(f'{code}下单成功，下单价格：{last_price}，下单量：{buy_volume}')
                record_log("下单时间：{}".format(datetime.datetime.now()))
                break
            else:
                record_log(f'{code}下单失败！')
                raise Exception(f'{code}下单失败！')
        except:
            pass

