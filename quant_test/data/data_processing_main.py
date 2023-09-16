# -*- coding: utf-8 -*-
"""
# ===注意事项
# 目前我们只根据市值选股，所以数据中只有一些基本数据加上市值。
# 实际操作中，会根据很多指标进行选股。在增加这些指标的时候，一定要注意在这两个函数中如何增加这些指标：
merge_with_index_data(), transfer_to_period_data()
# 比如增加：成交量、财务数据
"""

import platform
from multiprocessing import Pool, freeze_support, cpu_count
import warnings

from data.processing.factors_calculator import *
from data.processing.reformat_utils import *
from data.data_utils import *
from utils_global.dingding_message import *
from utils_global.global_config import *

warnings.filterwarnings('ignore')
# ===数据周期
period_type = 'W'  # W代表周，M代表月，D代表日
date_start = '2010-01-01'  # 回测开始时间
# date_end = '2023-07-07'  # 回测结束时间
date_end = get_current_date()  # 回测结束时间

# ===读取所有股票代码的列表
path = r'{}\data\historical\tushare_stock_data'.format(project_path)
stock_code_list = get_stock_code_list_in_one_dir(path)
stock_code_list = [code for code in stock_code_list if 'BJ' not in code]  # 排除北京证券交易所
# rule_out = ['301202.SZ', '301292.SZ', '301395.SZ', '301429.SZ', '301486.SZ', '301488.SZ', '688433.SH']  # 排除有问题的数据
rule_out = []
stock_code_list = [code for code in stock_code_list if code not in rule_out]

# ===循环读取并且合并
# 导入上证指数，保证指数数据和股票数据在同一天结束，不然会出现问题。
index_data = import_index_data(r'{}\data\historical\tushare_index_data\000001.SH.csv'.format(project_path),
                               back_trader_start=date_start, back_trader_end=date_end)


# 循环读取股票数据
def calculate_by_stock(code):
    """
    整理数据核心函数
    :param code: 股票代码
    :return: 一个包含该股票所有历史数据的DataFrame
    """
    try:
        # all_stock_data = pd.DataFrame()  # 用于存储数据
        # for code in stock_code_list:
        #     print(code)

        # 读入股票数据
        df = pd.read_csv(path + '/%s.csv' % code, encoding='gbk', parse_dates=['交易日期'])

        # 插入自然数索引，0是自然数
        df.reset_index(drop=True, inplace=True)
        # df['上市至今交易天数'] = df.index + 1

        # 计算按日的因子, extra_agg_dict在转换周期时使用
        extra_agg_dict, df = daily_factor_calculator(code, df, index_data)

        # =将日线数据转化为月线或者周线
        df = transfer_to_period_data(df, period_type=period_type, extra_agg_dict=extra_agg_dict)

        # =对数据进行整理
        # 删除上市的第一个周期
        df.drop([0], axis=0, inplace=True)  # 删除第一行数据
        # 删除2007年之前的数据
        df = df[df['交易日期'] > pd.to_datetime('20061215')]
        # 计算下周期每天涨幅
        df['下周期每天涨跌幅'] = df['每天涨跌幅'].shift(-1)
        df['下周期涨跌幅'] = df['本周期涨跌幅'].shift(-1)

        # 删除月末为st状态的周期数
        df = df[df['股票名称'].str.contains('ST') == False]
        # 删除月末为s状态的周期数
        df = df[df['股票名称'].str.contains('S') == False]
        # 删除月末有退市风险的周期数
        df = df[df['股票名称'].str.contains('\*') == False]
        df = df[df['股票名称'].str.contains('退') == False]
        # 删除月末不交易的周期数
        df = df[df['是否交易'] == 1]
        # 删除交易天数过少的周期数
        df = df[df['交易天数'] / df['市场交易天数'] >= 0.8]
        df.drop(['交易天数', '市场交易天数'], axis=1, inplace=True)
        print("{}周期 个股：{}数据处理完成".format(period_type, code))
        return df

    except Exception as e:
        # 在子进程中捕获异常，根据需要处理异常
        print(f"子进程出现异常, 任务: {code}, 异常: {e}")
        traceback.print_exc()
        return None  # 返回特定值或None表示处理异常


def parallel_data_processor(multiple_process=True):
    # 添加对windows多进程的支持
    # https://docs.python.org/zh-cn/3.7/library/multiprocessing.html
    if 'Windows' in platform.platform():
        freeze_support()

    # 标记开始时间
    if multiple_process:
        # 开始并行
        # 创建一个进程池
        pool = Pool(max(cpu_count() - 2, 1))
        # 任务列表
        tasks = sorted(stock_code_list)
        # 使用apply_async()来提交任务
        results = []
        for task in tasks:
            result = pool.apply_async(calculate_by_stock, args=(task,))
            results.append(result)

        # 关闭进程池
        pool.close()
        pool.join()

        # 获取子进程的返回值
        df_list = []
        for result in results:
            try:
                value = result.get()
                if value is not None:
                    df_list.append(value)
            except Exception as e:
                # 如果有异常，则在主线程中处理异常
                print("子进程异常被主线程捕获:", e)

    else:
        df_list = []
        for stock in stock_code_list:
            res_df = calculate_by_stock(stock)
            if res_df is not None:
                df_list.append(res_df)

    # 合并为一个大的DataFrame
    all_stock_data = pd.concat(df_list, ignore_index=True)
    all_stock_data.sort_values(['交易日期', '股票代码'], inplace=True)  # ===将数据存入数据库之前，先排序、reset_index
    all_stock_data.reset_index(inplace=True, drop=True)

    # 检查数据是否正确
    if dev_or_test:
        all_stock_data.tail(100).to_csv(r"{}\data\historical\processed_data\all_data_{}_tail100.csv".format(project_path, period_type), encoding='gbk')

    # 将数据存储到pickle文件
    all_stock_data.to_pickle(
        r'{}\data\historical\processed_data\all_stock_data_{}.pkl'.format(project_path, period_type))


if __name__ == '__main__':
    # calculate_by_stock('000001.SZ')
    try:
        print("开始处理{}数据".format(period_type))
        parallel_data_processor(multiple_process=True)
        print(period_type + "数据处理完成。")
        send_dingding("交易播报：执行 数据处理 成功！")
    except Exception as e:
        send_dingding("交易播报：执行 数据处理 失败，后续流程已阻断，请查看后台了解详细原因。")
        print(e)
