# -*- coding: utf-8 -*-
import pandas as pd


def query_single_order(xt_trader, user, order_id):
    # 查询单一订单状态
    single_entrust_res = xt_trader.query_stock_order(user, order_id)
    single_entrust = {'证券代码': single_entrust_res.stock_code, '下单价格': single_entrust_res.price,
                      '下单量': single_entrust_res.order_volume, '订单状态': single_entrust_res.order_status,
                      '订单标记': single_entrust_res.order_remark, '委托编号': single_entrust_res.order_id,
                      '成交均价': single_entrust_res.traded_price, '成交量': single_entrust_res.traded_volume,
                      '下单时间': single_entrust_res.order_time, '下单类型': single_entrust_res.order_type,
                      '价格类型': single_entrust_res.price_type}
    print(f'指定订单结果查询：{single_entrust}')
    return single_entrust


def query_order_list(xt_trader, user):
    # 查询所有订单状态
    all_entrust_res = xt_trader.query_stock_orders(user)
    entrust_list = []
    # 返回的委托数据需要逐个解析。
    for entrust in all_entrust_res:
        entrust_info = {'证券代码': entrust.stock_code, '下单价格': entrust.price, '下单量': entrust.order_volume,
                        '订单状态': entrust.order_status, '订单标记': entrust.order_remark, '委托编号': entrust.order_id,
                        '成交均价': entrust.traded_price, '成交量': entrust.traded_volume, '下单时间': entrust.order_time,
                        '下单类型': entrust.order_type, '价格类型': entrust.price_type}
        entrust_list.append(entrust_info)
    entrust_df = pd.DataFrame(entrust_list)
    print(f'今日所有订单订单结果查询：{entrust_df}')
    return entrust_df


def query_one_stock_position(xt_trader, user, code):
    # 查询单一股票持仓
    single_pos_res = xt_trader.query_stock_position(user, code)
    if single_pos_res:  # 若不持有指定的股票，返回None，不会进入if
        # 将查询到的持仓数据数据转为dict
        single_pos = {'证券代码': single_pos_res.stock_code, '成本价': single_pos_res.open_price, '持仓量': single_pos_res.volume,
                      '在途量': single_pos_res.on_road_volume, '可用量': single_pos_res.can_use_volume,
                      '冻结量': single_pos_res.frozen_volume, '昨日持仓量': single_pos_res.yesterday_volume,
                      '市值': single_pos_res.market_value}
        print(f'{code}当前持仓：{single_pos}')
    else:
        single_pos = {}
    return single_pos


def query_all_stock_position(xt_trader, user):
    # 查询所有股票持仓
    all_pos_res = xt_trader.query_stock_positions(user)
    if all_pos_res:  # 若账户空仓，返回None，不会进入if
        pos_list = []
        # 返回的持仓数据需要逐个解析。
        for pos in all_pos_res:
            pos_info = {'证券代码': pos.stock_code, '成本价': pos.open_price, '持仓量': pos.volume, '在途量': pos.on_road_volume,
                        '可用量': pos.can_use_volume, '冻结量': pos.frozen_volume, '昨日持仓量': pos.yesterday_volume,
                        '市值': pos.market_value}
            pos_list.append(pos_info)
        pos_df = pd.DataFrame(pos_list)
        print(f'当前所有持仓：{pos_df}')
    else:
        pos_df = pd.DataFrame()
    return pos_df


def query_stock_with_position(xt_trader, user):
    # 查询所有股票持仓
    all_pos_res = xt_trader.query_stock_positions(user)
    pos_dct = {}
    if all_pos_res:  # 若账户空仓，返回None，不会进入if
        # 返回的持仓数据需要逐个解析。
        for pos in all_pos_res:
            pos_dct[pos.stock_code] = pos.volume
        print(f'当前所有持仓股票代码：{pos_dct}')
    return pos_dct
