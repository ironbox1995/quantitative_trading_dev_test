# coding=gbk

from get_strategy_function import get_strategy_function
from backtest.repick_time import *
from back_test_config import *
from utils_global.global_config import *
import warnings
from utils_global.dingding_message import *

warnings.filterwarnings('ignore')


def back_test_latest_result(strategy_name, select_stock_num, period_type, alpha, pick_time_mtd=""):
    pick_stock_strategy = get_strategy_function(strategy_name)

    if not Second_Board_available:
        strategy_name += "无创业"
    if not STAR_Market_available:
        strategy_name += "无科创"

    print('策略名称:', strategy_name)
    print('周期:', period_type)
    print('择时方法:', pick_time_mtd)

    # 常量设置
    c_rate = 1 / 10000  # 手续费 这里与之前不同
    t_rate = 1 / 2000  # 印花税

    # ===导入数据
    # 从pickle文件中读取整理好的所有股票数据
    df = pd.read_pickle(
        r'{}\data\historical\processed_data\all_stock_data_{}.pkl'.format(project_path, period_type))
    # ===删除下个交易日不交易、开盘涨停的股票，因为这些股票在下个交易日开盘时不能买入。
    df = df[df['下日_是否交易'] == 1]
    df = df[df['下日_开盘涨停'] == False]
    df = df[df['下日_是否ST'] == False]
    df = df[df['下日_是否退市'] == False]

    # 选股策略，可通过导入不同的策略进行替换。
    session_id, df = pick_stock_strategy(df, select_stock_num)

    # ===整理选中股票数据
    # 挑选出选中股票
    df['股票代码'] += ' '
    df['股票名称'] += ' '
    group = df.groupby('交易日期')
    select_stock = pd.DataFrame()
    select_stock['股票数量'] = group['股票名称'].size()
    select_stock['买入股票代码'] = group['股票代码'].sum()
    select_stock['买入股票名称'] = group['股票名称'].sum()
    # 保存select_stock的最后一行
    latest_selection = select_stock.tail(1)

    # 去掉最后一行后再处理一次
    df.dropna(subset=['下周期每天涨跌幅'], inplace=True)
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

    # 根据资金曲线择时
    if pick_time_mtd == "" or pick_time_mtd == "无择时":
        pick_time_mtd = "无择时"
        latest_selection['最新择时信号'] = 1  # 即使无择时也要输出一个信号，便于正确读取。
    else:
        select_stock, latest_signal = pick_time(select_stock, pick_time_mtd)
        latest_selection['最新择时信号'] = latest_signal

    # 计算Q列的值
    select_stock['Q'] = select_stock['选股下周期涨跌幅'].copy()
    select_stock['Q'].iloc[1:] = 0
    for i in range(1, len(select_stock)):
        select_stock['Q'].iloc[i] = alpha * select_stock['选股下周期涨跌幅'].iloc[i] + (1 - alpha) * select_stock['Q'].iloc[i - 1]
    latest_Q = select_stock.tail(1)['Q'].iloc[0]
    latest_selection['Q'] = latest_Q
    select_stock['Q'] = select_stock['Q'].shift(1)
    select_stock['Q'].fillna(value=0, inplace=True)

    latest_selection.to_csv(
        r"{}\backtest\latest_selection\最新选股_{}_{}_选{}_{}.csv"
            .format(project_path, strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk')

    return select_stock


if __name__ == "__main__":

    for strategy_name in strategy_li:
        for period_type in period_type_li:
            for select_stock_num in select_stock_num_li:
                # pick_time_mtd = pick_time_mtd_dct[strategy_name]
                for pick_time_mtd in pick_time_li:
                    try:
                        back_test_latest_result(strategy_name, select_stock_num, period_type, ALPHA, pick_time_mtd)
                        # back_test_latest_result(strategy_name, select_stock_num, period_type, ALPHA, "无择时")
                    except Exception as e:
                        msg = "交易播报：策略{}结果输出失败：period_type:{}, select_stock_num:{}".format(strategy_name, period_type,
                                                                                    select_stock_num)
                        print(msg)
                        send_dingding(msg)
                        print(e)
    send_dingding("交易播报：执行 最新结果输出 成功！")
