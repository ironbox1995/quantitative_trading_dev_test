# 功能性函数
# 获取最新的交易价格
def cal_order_price(side, buy1_price, sell1_price, slippage, up_limit_price, down_limit_price):
    if side == 'sell':
        order_price = buy1_price * (1 - slippage)
        order_price = max(round(order_price, 2), down_limit_price)
    elif side == 'buy':
        order_price = sell1_price * (1 + slippage)
        order_price = min(round(order_price, 2), up_limit_price)
    else:
        raise ValueError('side参数必须是 buy 或者 sell')

    return order_price