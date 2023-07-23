"""
从tushare获取数据
# # 获取利润表数据
# df_income = pro.income(ts_code=stock_code, start_date=start_date, end_date=end_date)
# # 资产负债表数据
# df_balance = pro.balancesheet(ts_code=stock_code, start_date=start_date, end_date=end_date)
# # 现金流数据
# df_cash_flow = pro.cashflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
"""
from processing.reformat_utils import *
import time


def get_index_data_from_tushare(pro, stock_code, start_date, end_date):
    """
    https://tushare.pro/document/2
    代码部分来自chatGPT
    :param stock_code:股票代码
    :param start_date:起始日期
    :param end_date:结束日期
    # 日期格式
    # start_date = '20200101'  # 起始日期
    # end_date = '20210331'  # 结束日期
    :return:
    """

    # 获取指数历史行情数据
    one_index_data_tushare = pro.index_daily(ts_code=stock_code, start_date=start_date, end_date=end_date)

    one_index_data_tushare['trade_date'] = pd.to_datetime(one_index_data_tushare['trade_date'])
    one_index_data_tushare.sort_values(by=['trade_date'], ascending=True, inplace=True)

    # 重命名列, 部分数据的单位可能与后续所需不同，但是应在处理数据时解决
    rename_dict = {'ts_code': '指数代码', 'trade_date': '交易日期', 'open': '指数开盘价', 'close': '指数收盘价', 'high': '指数最高价',
                   'low': '指数最低价', 'vol': '指数成交量', 'amount': '指数成交额', 'pct_chg': '指数涨跌幅', 'change': '指数涨跌额',
                   'pre_close': '指数前收盘价'}
    one_index_data_tushare.rename(columns=rename_dict, inplace=True)

    return one_index_data_tushare


def update_and_save_stock_data(pro, start_date, end_date):
    """
    获取并保存数据
    :param pro: 接口
    :param start_date:起始日期
    :param end_date:结束日期
    :return:
    """
    # 仅返回上市的
    index_code_li = ['000001.SH', '399001.SZ', '399006.SZ', '000688.SH', '000300.SH', '000016.SH', '000905.SH',
                  '000852.SH']  # 上证指数、深证成指、创业板指、科创50、沪深300、上证50、中证500、中证1000
    print("开始从tushare下载历史指数数据，本次开始下载的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for index_code in index_code_li:
        try:
            one_index_data_tushare = get_index_data_from_tushare(pro, index_code, start_date, end_date)
            # 构建存储文件路径
            path = 'F:/quantitative_trading_dev_test/quant_test/data/historical/tushare_index_data/' + index_code + '.csv'
            # 保存数据
            save_data(path, "指数日K线", index_code, one_index_data_tushare)
            time.sleep(0.2)
        except Exception as e:
            print(index_code + "指数数据获取失败：", e)

    print("从tushare下载历史k线数据结束，本次下载结束的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


def get_historical_index_data_main(index_path):
    if index_path == "":
        latest_date = "20070101"
    elif os.path.exists(index_path):
        latest_date = pd.read_csv(index_path, encoding='gbk').tail(1)['交易日期'].values[0]
        latest_date = parse_update_start_time(latest_date)
    else:
        latest_date = "20070101"

    # 设置tushare的token，可以在tushare官网（https://tushare.pro/）申请免费token
    ts.set_token('30e6c0329269ab3e3ac6dfcc8737b274084e683ea121395597940bcc')

    # 初始化tushare pro接口
    pro = ts.pro_api()

    update_and_save_stock_data(pro, start_date=latest_date, end_date=get_today())


if __name__ == "__main__":
    index_path = r"F:\quantitative_trading_dev_test\quant_test\data\historical\tushare_index_data\000001.SH.csv"
    get_historical_index_data_main(index_path)
