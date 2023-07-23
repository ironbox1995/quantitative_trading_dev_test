# 从QMT获取历史数据
from xtquant import xtdata
from tqdm import tqdm
from processing.reformat_utils import *


def get_shse_a_list():
    """
    获取沪深指数、所有A股、ETF列表
    """
    # 上证指数、深证成指、创业板指、科创50、沪深300、上证50、中证500、中证1000、沪深300
    index_code = ['000001.SH', '399001.SZ', '399006.SZ', '000688.SH', '000300.SH', '000016.SH', '000905.SH',
                  '000852.SH', '399300.SZ']
    a_code = xtdata.get_stock_list_in_sector('沪深A股')
    etf_code = xtdata.get_stock_list_in_sector('沪深ETF')

    return index_code + a_code + etf_code, a_code


def download_historical_kline(start_time='', period='1d'):
    """
    下载历史K线数据，路径：userdata_mini\datadir
    调整start_time即可变为更新数据
    """
    code_list, a_list = get_shse_a_list()
    print("开始下载历史k线数据，本次开始下载的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for code in tqdm(code_list):
        xtdata.download_history_data(code, period=period, start_time=start_time)
    print("下载历史k线数据结束，本次下载结束的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


def reformat_historical_kline(start_time='', period='1d'):
    """
    修改k线数据格式
    :param period:
    :return:
    """
    code_list, a_list = get_shse_a_list()
    data_ts = get_basic_info_from_tushare()
    for code in tqdm(code_list):
        data = xtdata.get_local_data(field_list=[], stock_code=[code], period=period, start_time=start_time)
        data_df = reformat_data_one_stock(data,data_ts, code)

        # 构建存储文件路径
        path = './historical/QMT_stock_data/' + code + '.csv'

        # 保存数据
        save_data(path, "K线", code, data_df)


def download_historical_financial_data(start_time=''):
    """
    下载历史财务数据
    调整start_time即可变为更新数据
    """
    code_list, a_list = get_shse_a_list()
    table_list = ['Balance','Income','CashFlow','Capital','Top10FlowHolder','Top10Holder','HolderNum','PreshareIndex']
    print("开始下载历史财务数据，本次开始下载的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for code in tqdm(a_list):
        xtdata.download_financial_data([code], table_list=table_list, start_time=start_time)
    print("下载历史财务数据结束，本次下载结束的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


def reformat_historical_financial_data(start_time=''):
    """
    修改财务数据格式
    :return:
    """
    code_list, a_list = get_shse_a_list()
    table_list = ['Balance','Income','CashFlow','Capital','Top10FlowHolder','Top10Holder','HolderNum','PreshareIndex']

    print("正在增加或更新财务数据：")
    for code in tqdm(a_list):  # 仅抽取a股财务数据
        fin_data = xtdata.get_financial_data([code], table_list=table_list, start_time=start_time)
        for table in table_list:
            fin_df = pd.DataFrame(fin_data[code][table])

            # 构建存储文件路径
            path = './historical/financial_data/' + code + '-' + table + '.csv'

            # 保存数据
            fin_df.to_csv(path, index=False, mode='w', encoding='gbk')


def get_historical_data_main(index_path):

    if index_path == "":
        latest_date = "20070101"
    elif os.path.exists(index_path):
        latest_date = pd.read_csv(index_path, encoding='gbk').tail(1)['交易日期'].values[0]
        latest_date = parse_update_start_time(latest_date)
    else:
        latest_date = "20070101"

    print("开始更新数据，起始日期：{}".format(latest_date))
    print("开始更新历史K线数据。")
    download_historical_kline(start_time=latest_date)
    print("更新历史K线数据结束。")
    print("开始更新历史财务数据")
    download_historical_financial_data(start_time=latest_date)
    print("更新历史财务数据结束")


def reformat_historical_data_main(index_path):
    if os.path.exists(index_path):
        latest_date = pd.read_csv(index_path, encoding='gbk').tail(1)['交易日期'].values[0]
        latest_date = parse_update_start_time(latest_date)
    else:
        latest_date = "20070101"

    print("开始重构历史K线数据。")
    reformat_historical_kline(start_time=latest_date)
    print("重构历史K线数据结束。")
    print("开始重构历史财务数据。")
    reformat_historical_financial_data(start_time=latest_date)
    print("重构历史财务数据结束。")


if __name__ == "__main__":
    index_path = r"F:\quantitative_trading_dev_test\quant_test\data\historical\QMT_stock_data\000001.SH.csv"
    get_historical_data_main(index_path)
    reformat_historical_data_main(index_path)
