"""
从tushare获取数据
# # 获取利润表数据
# df_income = pro.income(ts_code=stock_code, start_date=start_date, end_date=end_date)
# # 资产负债表数据
# df_balance = pro.balancesheet(ts_code=stock_code, start_date=start_date, end_date=end_date)
# # 现金流数据
# df_cash_flow = pro.cashflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
"""
from data.processing.reformat_utils import *
from Config.global_config import *
from data.data_utils import *
import time
import traceback


def get_stock_data_from_tushare(pro, stock_code, stock_basic_info, industry_dct, start_date, end_date):
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

    # 获取股票历史行情数据
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    # 调用接口获取历史指标数据
    df_basic = pro.daily_basic(ts_code=stock_code, start_date=start_date, end_date=end_date)

    one_stock_data_tushare = pd.merge(df, df_basic)

    # 查询股票基本信息，包含一些名称之类的
    one_stock_data_tushare['股票名称'] = stock_basic_info['name']
    one_stock_data_tushare['行业'] = stock_basic_info['industry']
    one_stock_data_tushare['市场类型'] = stock_basic_info['market']
    one_stock_data_tushare['申万三级行业'] = industry_dct['申万三级行业']
    one_stock_data_tushare['申万二级行业'] = industry_dct['申万二级行业']
    one_stock_data_tushare['申万一级行业'] = industry_dct['申万一级行业']

    one_stock_data_tushare['trade_date'] = pd.to_datetime(one_stock_data_tushare['trade_date'])
    one_stock_data_tushare.sort_values(by=['trade_date'], ascending=True, inplace=True)

    # 重命名列, 部分数据的单位可能与后续所需不同，但是应在处理数据时解决
    rename_dict = {'ts_code': '股票代码', 'trade_date': '交易日期', 'open': '开盘价', 'close': '收盘价', 'high': '最高价',
                   'low': '最低价', 'vol': '成交量', 'amount': '成交额', 'pct_chg': '涨跌幅', 'change': '涨跌额',
                   'pre_close': '前收盘价', 'turnover_rate': '换手率（%）', 'turnover_rate_f': '换手率（自由流通股）',
                   'volume_ratio': '量比', 'pe': '市盈率', 'pe_ttm': '市盈率TTM', 'pb': '市净率', 'ps': '市销率',
                   'ps_ttm': '市销率TTM', 'dv_ratio': '股息率（%）', 'dv_ttm': '股息率TTM（%）', 'total_share': '总股本 （万股）',
                   'float_share': '流通股本 （万股）', 'free_share': '自由流通股本 （万）', 'total_mv': '总市值 （万元）',
                   'circ_mv': '流通市值（万元）'}
    one_stock_data_tushare.rename(columns=rename_dict, inplace=True)
    one_stock_data_tushare = one_stock_data_tushare[['股票代码', '股票名称', '交易日期', '行业', '申万一级行业', '申万二级行业',
                                                     '申万三级行业', '市场类型', '开盘价', '最高价', '最低价', '收盘价', '前收盘价',
                                                     '涨跌额', '涨跌幅', '成交量', '成交额', '换手率（%）', '换手率（自由流通股）',
                                                     '量比', '市盈率', '市盈率TTM', '市净率', '市销率', '市销率TTM',
                                                     '股息率（%）', '股息率TTM（%）', '总股本 （万股）', '流通股本 （万股）',
                                                     '自由流通股本 （万）', '总市值 （万元）', '流通市值（万元）']]

    return one_stock_data_tushare


def update_and_save_stock_data(pro, start_date, end_date):
    """
    获取并保存数据
    :param pro: 接口
    :param start_date:起始日期
    :param end_date:结束日期
    :return:
    """
    # 仅返回上市的
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,market,industry,'
                                                                         'list_date')
    df_industry = pd.read_csv(r'{}\data\historical\tushare_industry_data\industry_data.csv'.format(project_path))

    print("开始从tushare下载历史k线数据，本次开始下载的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    print("下载数据起止时间为：{}-{}".format(start_date, end_date))
    for i in range(len(data)):
        try:
            stock_code = data.iloc[i]['ts_code']
            if 'BJ' in stock_code:
                print("不下载北京证券交易所股票数据：{}".format(stock_code))
                continue
            stock_basic_info = data.iloc[i]  # series类型

            # 获取行业数据
            industry_dct = df_industry[df_industry["con_code"] == stock_code].to_dict(orient='list')
            industry_dct = reorganize_industry_dct(industry_dct)

            one_stock_data_tushare = get_stock_data_from_tushare(pro, stock_code, stock_basic_info, industry_dct,
                                                                 start_date, end_date)
            # 构建存储文件路径
            path = r'{}\data\historical\tushare_stock_data\{}.csv'.format(project_path, stock_code)
            # 保存数据
            save_data(path, "日K线", stock_code, one_stock_data_tushare)
            time.sleep(0.2)
        except Exception as e:
            stock_code = data.iloc[i]['ts_code']
            print(stock_code + "数据获取失败：", e)
            traceback.print_exc()

    print("从tushare下载历史k线数据结束，本次下载结束的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


def get_tushare_historical_kline_data_main(index_path):
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
    index_path = r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path)
    get_tushare_historical_kline_data_main(index_path)
