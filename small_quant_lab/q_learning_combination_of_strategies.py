# 思路：使用Q学习中的eps-greedy方法并联策略，尝试寻找最优解，为防止所有策略同时失效的可能性，应加入空仓策略
import random
from data.processing.Functions import *
from backtest.repick_time import *
from backtest.latest_result import back_test_latest_result
from Config.back_test_config import *
from backtest.Evaluate import *


def create_empty_strategy(pick_time_switch):
    # 导入指数数据
    index_data = import_index_data(
        r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path)
        , back_trader_start=date_start, back_trader_end=date_end)
    # 创造空的事件周期表，用于填充不选股的周期
    empty_df = create_empty_data(index_data, 'W')
    empty_df['Q'] = 0
    if pick_time_switch:
        empty_df['signal'] = 1.0
        empty_df = empty_df[['股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅', 'signal', 'Q']]
    else:
        empty_df = empty_df[['股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅', 'Q']]
    return empty_df


def combine_all_strategies(eps, alpha, pick_time_switch):
    period_type = 'W'
    select_stock_num = 3

    strategy_dct = {}

    for strategy_name in strategy_li:
        if pick_time_switch:
            select_stock_single = back_test_latest_result(strategy_name, select_stock_num, period_type, alpha, pick_time_mtd=pick_time_mtd_dct[strategy_name])
        else:
            select_stock_single = back_test_latest_result(strategy_name, select_stock_num, period_type, alpha, pick_time_mtd="无择时")
        strategy_dct[strategy_name] = select_stock_single

    strategy_dct['空仓策略'] = create_empty_strategy(pick_time_switch)

    for k, v in strategy_dct.items():
        v = v[v.index > pd.to_datetime('20100201')]  # 不知道为什么数据参差不齐 TODO: 数据对齐
        v.reset_index(inplace=True, drop=False)  # 还原index，作为一列
        strategy_dct[k] = v

    select_stock = pd.DataFrame(columns=['交易日期', '股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅', 'Q'])

    for i in range(len(strategy_dct['空仓策略'])):
        max_q = 0
        max_dataframe = None
        strategy_df_to_chose_from = []
        for strategy_df in strategy_dct.values():
            q_value = strategy_df.iloc[i]['Q']
            if pick_time_switch:
                signal = strategy_df.iloc[i]['signal']
                if signal == 1.0:
                    strategy_df_to_chose_from.append(strategy_df)
                if q_value > max_q and signal == 1.0:
                    max_q = q_value
                    max_dataframe = strategy_df
            else:
                if q_value > max_q:
                    max_q = q_value
                    max_dataframe = strategy_df

        if max_dataframe is None:
            max_dataframe = strategy_dct['空仓策略']

        random_float = random.uniform(0, 1)
        if random_float < eps:
            chosen_dataframe = random.choice(strategy_df_to_chose_from)  # 从信号为1的策略中随机选择
        else:
            chosen_dataframe = max_dataframe
        selected_row = chosen_dataframe.iloc[i]
        select_stock = select_stock.append(selected_row, ignore_index=True)
        # print(len(select_stock))

    strategy_name = 'Q学习并联策略'

    # 导入指数数据
    index_data = import_index_data(
        r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path)
        , back_trader_start=date_start, back_trader_end=date_end)

    # 创造空的事件周期表，用于填充不选股的周期
    empty_df = create_empty_data(index_data, period_type)
    empty_df['Q'] = 0

    # 计算整体资金曲线
    # select_stock.reset_index(inplace=True)
    select_stock['资金曲线'] = (select_stock['选股下周期涨跌幅'] + 1).cumprod()
    select_stock.set_index('交易日期', inplace=True)
    empty_df.update(select_stock)
    empty_df['资金曲线'] = select_stock['资金曲线']
    select_stock = empty_df
    select_stock.reset_index(inplace=True, drop=False)

    pick_time_mtd = "有择时" if pick_time_switch else "无择时"
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

    equity['涨跌幅'] = select_stock['选股下周期每天涨跌幅'].sum()  # 累加是没错的
    equity['equity_curve'] = (equity['涨跌幅'] + 1).cumprod()
    equity['benchmark'] = (equity['指数涨跌幅'] + 1).cumprod()

    # pick_time_mtd = "有择时" if pick_time_switch else "无择时"
    equity.to_csv(r"{}\backtest\result_record\equity_{}_{}_选{}_{}-{}_{}.csv"
                  .format(project_path, strategy_name, period_type, select_stock_num, date_start, date_end, pick_time_mtd),
                  encoding='gbk')

    serial_number = generate_serial_number()
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
        print('参数：eps = {}，alpha = {}'.format(eps, alpha), file=f)
        print('回测开始时间：{}，回测结束时间：{}'.format(date_start, date_end), file=f)
        print(rtn, file=f)
        print("=" * 30, file=f)
        print("", file=f)

    # ===画图
    equity = equity.reset_index()
    # pick_time_mtd = "有择时" if pick_time_switch else "无择时"
    draw_equity_curve_mat(equity, data_dict={'策略表现': 'equity_curve', '基准涨跌幅': 'benchmark'}, date_col='交易日期'
                          , strategy_name=strategy_name, period_type=period_type, select_stock_num=select_stock_num
                          , serial_number=serial_number, show_pic=False, pick_time_mtd=pick_time_mtd)


if __name__ == "__main__":
    # eps_li = [-1]
    # alpha_li = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    # pick_time_switch = True
    # for e in eps_li:
    #     for a in alpha_li:
    #         print('参数：eps = {}，alpha = {}'.format(e, a))
    #         combine_all_strategies(eps=e, alpha=a, pick_time_switch)
    pick_time_switch = False
    combine_all_strategies(eps, ALPHA, pick_time_switch)
