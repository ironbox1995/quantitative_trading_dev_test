from data.processing.reformat_utils import *
import time

from Config.global_config import *


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


def update_and_save_index_data(pro, start_date, end_date):
    """
    获取并保存数据
    :param pro: 接口
    :param start_date:起始日期
    :param end_date:结束日期
    :return:
    """
    # 上证综指、深证成指、创业板指、科创50、沪深300、上证50、中证500、中证1000、中证2000、中证800
    index_code_li = ['000001.SH', '399001.SZ', '399006.SZ', '000688.SH', '000300.SH', '000016.SH', '000905.SH',
                     '000852.SH', '932000.CSI', '000906.SH']
    # 行业指数（需要tushare5000积分）
    # sw_l1_index = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI', '801060.SI', '801070.SI', '801080.SI', '801090.SI', '801100.SI', '801110.SI', '801120.SI', '801130.SI', '801140.SI', '801150.SI', '801160.SI', '801170.SI', '801180.SI', '801190.SI', '801200.SI', '801210.SI', '801220.SI', '801230.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI', '801750.SI', '801760.SI', '801770.SI', '801780.SI', '801790.SI', '801880.SI', '801890.SI']
    # index_code_li += sw_l1_index

    print("开始从tushare下载历史指数数据，本次开始下载的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    for index_code in index_code_li:
        try:
            one_index_data_tushare = get_index_data_from_tushare(pro, index_code, start_date, end_date)
            # 构建存储文件路径
            path = r'{}\data\historical\tushare_index_data\{}.csv'.format(project_path, index_code)
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

    update_and_save_index_data(pro, start_date=latest_date, end_date=get_today())


if __name__ == "__main__":
    index_path = r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path)
    get_historical_index_data_main(index_path)
