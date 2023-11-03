# -*- coding: utf-8 -*-
"""
《邢不行-2021新版|Python股票量化投资课程》
author: 邢不行
微信: xbx9585

根据选股数据，进行选股
我粗略想了想，如果不考虑均仓买入后剩余的资金的量的话，根本就不需要考虑总共有多少钱。所以这样的回测其实是合理的。
"""
from backtest.Evaluate import *
from data.processing.Functions import *
from strategy.get_strategy_function import get_pick_stock_strategy
from backtest.repick_time import *
from backtest.utils import *
from Config.back_test_config import *
from Config.global_config import *
import warnings
import traceback
from utils_global.dingding_message import *

import datetime

warnings.filterwarnings('ignore')

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数


def back_test_main(df, index_data, strategy_name, date_start, date_end, select_stock_num, period_type, serial_number
                   , pick_time_mtd="", show_pic=False):
    # 策略选择
    pick_stock_strategy = get_pick_stock_strategy(strategy_name)

    if not Second_Board_available:
        strategy_name += "无创业"
    if not STAR_Market_available:
        strategy_name += "无科创"

    print('策略名称:', strategy_name)
    print('周期:', period_type)
    print('再择时方法:', pick_time_mtd)

    # 常量设置
    c_rate = 1 / 10000  # 手续费 这里与之前不同
    t_rate = 1 / 2000  # 印花税

    # 创造空的事件周期表，用于填充不选股的周期
    empty_df = create_empty_data(index_data, period_type)

    # 处理数据
    df.dropna(subset=['下周期每天涨跌幅'], inplace=True)
    # ===删除下个交易日不交易、开盘涨停的股票，因为这些股票在下个交易日开盘时不能买入。
    df = df[df['下日_是否交易'] == 1]
    df = df[df['下日_开盘涨停'] == False]
    df = df[df['下日_是否ST'] == False]
    df = df[df['下日_是否退市'] == False]

    # 选股策略，可通过导入不同的策略进行替换。
    session_id, df = pick_stock_strategy(df, select_stock_num)

    # ===按照开盘买入的方式，修正选中股票在下周期每天的涨跌幅。
    df['下日_开盘买入涨跌幅'] = df['下日_开盘买入涨跌幅'].apply(lambda x: [x])
    df['下周期每天涨跌幅'] = df['下周期每天涨跌幅'].apply(lambda x: x[1:])
    df['下周期每天涨跌幅'] = df['下日_开盘买入涨跌幅'] + df['下周期每天涨跌幅']

    # ===整理选中股票数据
    # 挑选出选中股票
    df['股票代码'] += ' '
    df['股票名称'] += ' '
    group = df.groupby('交易日期')
    select_stock = pd.DataFrame()
    select_stock['股票数量'] = group['股票名称'].size()
    select_stock['买入股票代码'] = group['股票代码'].sum()
    select_stock['买入股票名称'] = group['股票名称'].sum()

    # 计算下周期每天的资金曲线，对几只股票取平均
    select_stock['选股下周期每天资金曲线'] = group['下周期每天涨跌幅'].apply(
        lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))
    # print(np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))  # 连乘，计算资金曲线，对几只股票取平均

    # 扣除买入手续费
    select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'] * (1 - c_rate)  # 计算有不精准的地方
    # 扣除卖出手续费、印花税。最后一天的资金曲线值，扣除印花税、手续费
    select_stock['选股下周期每天资金曲线'] = select_stock['选股下周期每天资金曲线'].apply(
        lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])

    # 计算下周期整体涨跌幅
    select_stock['选股下周期涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(lambda x: x[-1] - 1)
    # 计算下周期每天的涨跌幅
    select_stock['选股下周期每天涨跌幅'] = select_stock['选股下周期每天资金曲线'].apply(
        lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
    del select_stock['选股下周期每天资金曲线']

    # 计算整体资金曲线
    select_stock.reset_index(inplace=True)
    select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
    select_stock.set_index('交易日期', inplace=True)
    empty_df.update(select_stock)
    empty_df['资金曲线'] = select_stock['资金曲线']
    select_stock = empty_df
    select_stock.reset_index(inplace=True, drop=False)

    # 根据资金曲线择时
    if pick_time_mtd == "" or pick_time_mtd == "无择时":
        pick_time_mtd = "无择时"
    else:
        select_stock, latest_signal = curve_pick_time(select_stock, pick_time_mtd, index_data)

    select_stock.to_csv(
        r"{}\backtest\result_record\select_stock_{}_{}_选{}_{}-{}_{}.csv"
        .format(project_path, strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd), encoding='gbk')

    # ===计算选中股票每天的资金曲线
    # 计算每日资金曲线
    equity = pd.merge(left=index_data, right=select_stock[['交易日期', '买入股票代码']], on=['交易日期'],
                      how='left', sort=True)  # 将选股结果和大盘指数合并
    # equity.to_csv("equity.csv", encoding='utf_8_sig')

    equity['持有股票代码'] = equity['买入股票代码'].shift()
    equity['持有股票代码'].fillna(method='ffill', inplace=True)
    equity.dropna(subset=['持有股票代码'], inplace=True)
    # del equity['买入股票代码']

    equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()
    equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
    equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()

    equity.to_csv(r"{}\backtest\result_record\equity_{}_{}_选{}_{}-{}_{}.csv"
                  .format(project_path, strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd),
                  encoding='gbk')

    # ===计算策略评价指标
    rtn, year_return, month_return, latest_drawdown = strategy_evaluate(equity, select_stock)
    with open(r"{}\backtest\result_record\策略执行日志.txt".format(project_path), 'a',
              encoding='utf-8') as f:
        print("=" * 30, file=f)
        print(serial_number, file=f)
        print('回测执行时间：{}'.format(datetime.datetime.now()), file=f)
        print('策略名称: {}'.format(strategy_name), file=f)
        print('选股周期:{}'.format(period_type), file=f)
        print('选股数量:{}'.format(select_stock_num), file=f)
        print('择时方法:{}'.format(pick_time_mtd), file=f)
        print('回测开始时间：{}，回测结束时间：{}'.format(date_start, date_end), file=f)
        print(rtn, file=f)
        print('最近一次回撤幅度:{}'.format(latest_drawdown), file=f)
        print("=" * 30, file=f)
        print("", file=f)

    # 保存最近一次回撤
    latest_drawdown_df = pd.DataFrame()
    latest_drawdown_df['最近回撤幅度'] = latest_drawdown
    latest_drawdown_df.to_csv(
        r"{}\backtest\latest_selection\最近回撤_{}_{}_选{}_{}.csv"
            .format(project_path, strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk')


    # ===画图
    equity = equity.reset_index()
    draw_equity_curve_mat(equity, data_dict={'策略表现': 'equity_curve', '基准涨跌幅': 'benchmark'}, date_col='交易日期'
                          , strategy_name=strategy_name, period_type=period_type, select_stock_num=select_stock_num
                          , serial_number=serial_number, show_pic=show_pic, pick_time_mtd=pick_time_mtd)


if __name__ == "__main__":
    index_data = import_index_data(r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path), back_trader_start=date_start, back_trader_end=date_end)

    # ==============批量回测==============
    for period_type in period_type_li:
        df = pd.read_pickle(r'{}\data\historical\processed_data\all_stock_data_{}.pkl'.format(project_path, period_type))
        for strategy_name in strategy_li:
            for select_stock_num in select_stock_num_li:
                pick_time_mtd = pick_time_mtd_dct[strategy_name]
                # for pick_time_mtd in pick_time_li:
                try:
                    serial_number = generate_serial_number()
                    back_test_main(df, index_data, strategy_name, date_start, date_end, select_stock_num, period_type, serial_number,
                                   pick_time_mtd)
                    # serial_number = generate_serial_number()
                    # back_test_main(df, index_data, strategy_name, date_start, date_end, select_stock_num, period_type, serial_number,
                    #                "无择时")
                except Exception as e:
                    msg = "交易播报：策略 {} 执行失败：".format(strategy_name)
                    print(msg)
                    send_dingding(msg)
                    print(e)
                    traceback.print_exc()
    send_dingding("交易播报：执行 回测 成功！")
