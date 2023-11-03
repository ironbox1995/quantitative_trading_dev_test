# -*- coding: utf-8 -*-
from data.get_financial_data_from_tushare import *
from data.get_index_data_from_tushare import *
from data.get_stock_data_from_tushare import *
from data.get_industry_data_from_tushare import *
from utils_global.dingding_message import *
from Config.global_config import *

import traceback

def update_data_main():
    tushare_index_path = r"{}\data\historical\tushare_index_data\000001.SH.csv".format(project_path)
    # if os.path.exists(tushare_index_path):
    #     latest_date = pd.read_csv(tushare_index_path, encoding='gbk').tail(1)['交易日期'].values[0]
    #     latest_date = parse_update_start_time(latest_date)
    # else:
    #     latest_date = "20070101"

    print("开始下载tushare行业数据。")
    get_tushare_industry_data_main()
    print("tushare行业数据下载完成。")

    print("开始下载tusharek线数据。")
    get_tushare_historical_kline_data_main(tushare_index_path)
    print("tusharek线数据下载完成。")

    print("开始下载tushare指数数据。")
    get_historical_index_data_main(tushare_index_path)
    print("tushare指数数据下载完成。")

    print("开始下载tushare财务数据。")
    get_tushare_financial_data_main()
    print("tushare财务数据下载完成。")

    # ============QMT数据下载============
    # try:
    #     print("开始下载并更新财务数据。")
    #     download_historical_financial_data(start_time=latest_date)
    #     reformat_historical_financial_data(start_time=latest_date)
    # except Exception as e:
    #     print("更新财务数据失败。请检查是否打开miniQMT客户端。", e)


if __name__ == "__main__":
    try:
        update_data_main()
        send_dingding("交易播报：执行 数据获取 成功！")
    except Exception as e:
        send_dingding("交易播报：执行 数据获取 失败，后续流程已阻断，请查看后台了解详细原因。")
        traceback.print_exc()
        print(e)
