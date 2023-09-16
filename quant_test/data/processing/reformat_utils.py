# from xtquant import xtdata
import pandas as pd
import datetime
import os
import tushare as ts


def reformat_data_one_stock(data, data_ts, code):
    """
    将数据中的下面这些内容合并为一个DataFrame
    # dict_keys(['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag'])
            "time"                #时间戳
            "open"                #开盘价
            "high"                #最高价
            "low"                 #最低价
            "close"               #收盘价
            "volume"              #成交量
            "amount"              #成交额
            "settle"              #今结算
            "openInterest"        #持仓量
    处理成：股票代码,股票名称,交易日期,开盘价,最高价,最低价,收盘价,前收盘价,成交量,成交额  的格式
    :param data: 待处理数据
    :param code: 对应的股票代码
    :return:
    """
    columes_for_one_stock = []

    df_basic = data['time'].T
    df_basic['交易日期'] = df_basic[code].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    del df_basic[code]
    df_basic['股票代码'] = code
    df_basic = df_basic[['股票代码', '交易日期']]
    if len(data_ts[data_ts["ts_code"] == code]) > 0:
        df_basic['股票名称'] = data_ts[data_ts["ts_code"] == code].iloc[0]['name']
        df_basic['行业'] = data_ts[data_ts["ts_code"] == code].iloc[0]['industry']
        df_basic['市场类型'] = data_ts[data_ts["ts_code"] == code].iloc[0]['market']
    else:
        df_basic['股票名称'] = '-1'
        df_basic['行业'] = '-1'
        df_basic['市场类型'] = '-1'
    columes_for_one_stock.append(df_basic)

    df_open = data['open'].T
    df_open.rename(columns={code: '开盘价'}, inplace=True)
    columes_for_one_stock.append(df_open)

    df_high = data['high'].T
    df_high.rename(columns={code: '最高价'}, inplace=True)
    columes_for_one_stock.append(df_high)

    df_low = data['low'].T
    df_low.rename(columns={code: '最低价'}, inplace=True)
    columes_for_one_stock.append(df_low)

    df_close = data['close'].T
    df_close.rename(columns={code: '收盘价'}, inplace=True)
    columes_for_one_stock.append(df_close)

    df_preClose = data['preClose'].T
    df_preClose.rename(columns={code: '前收盘价'}, inplace=True)
    columes_for_one_stock.append(df_preClose)

    df_volume = data['volume'].T
    df_volume.rename(columns={code: '成交量'}, inplace=True)
    columes_for_one_stock.append(df_volume)

    df_amount = data['amount'].T
    df_amount.rename(columns={code: '成交额'}, inplace=True)
    columes_for_one_stock.append(df_amount)

    # 这两行似乎没有数据，结果全是0
    # df_settelementPrice = data['settelementPrice'].T
    # df_settelementPrice.rename(columns={code: '今结算'}, inplace=True)
    # columes_for_one_stock.append(df_settelementPrice)
    #
    # df_openInterest = data['openInterest'].T
    # df_openInterest.rename(columns={code: '持仓量'}, inplace=True)
    # columes_for_one_stock.append(df_openInterest)

    one_stock_data = pd.concat(columes_for_one_stock, axis=1)
    return one_stock_data


def parse_update_start_time(input_date_str):
    """
    获取更新的起始日期 by CHATGPT
    :param input_date_str: %Y-%m-%d 格式
    :return: %Y%m%d 格式 并且加一天
    """
    # 将输入的日期字符串转换成 datetime 对象
    input_date = datetime.datetime.strptime(input_date_str, '%Y-%m-%d')

    # 加一天
    output_date = input_date + datetime.timedelta(days=1)

    # 将输出的日期对象转换成字符串
    output_date_str = output_date.strftime('%Y%m%d')

    return output_date_str


def get_today():
    # 获取今日日期
    today = datetime.datetime.now().date()

    # 将日期格式化为字符串，格式为"YYYYMMDD"
    today_date = today.strftime("%Y%m%d")

    return today_date


def deduplicate_csv(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path, encoding='gbk')

    # 按日期去重
    if "交易日期" in df.columns:
        df.drop_duplicates(subset=["交易日期"], inplace=True)
    else:
        df.drop_duplicates(subset=["披露日期"], inplace=True)

    # 将结果写回 CSV 文件
    df.to_csv(file_path, index=False, encoding='gbk')


def save_data(path, data_type, code, df_to_save):
    try:
        # 文件存在，不是新股
        if os.path.exists(path):
            latest_date = pd.read_csv(path, encoding='gbk').tail(1)['交易日期'].values[0]
            if len(df_to_save) > 0 and latest_date != df_to_save.iloc[-1]['交易日期']:
                df_to_save.to_csv(path, header=None, index=False, mode='a', encoding='gbk')
                print("更新{}数据：{}".format(data_type, code))
            else:
                print("{}数据已是最新：{}".format(data_type, code))
        # 文件不存在，说明是新股
        else:
            df_to_save.to_csv(path, index=False, mode='a', encoding='gbk')
            print("新增{}数据：{}".format(data_type, code))
        # 对文件进行按行去重
        deduplicate_csv(path)

    except Exception as e:
        print(e)
        print("处理失败，文件路径为：{}".format(path))


def save_financial_data(path, data_type, code, df_to_save):
    try:
        # 文件存在，不是新股
        if os.path.exists(path):
            latest_date = pd.read_csv(path, encoding='gbk').tail(1)['披露日期'].values[0]
            if len(df_to_save) > 0 and latest_date != df_to_save.iloc[-1]['披露日期']:
                df_to_save.to_csv(path, header=None, index=False, mode='a', encoding='gbk')
                print("更新{}数据：{}".format(data_type, code))
            else:
                print("{}数据已是最新：{}".format(data_type, code))
        # 文件不存在，说明是新股
        else:
            df_to_save.to_csv(path, index=False, mode='a', encoding='gbk')
            print("新增{}数据：{}".format(data_type, code))
        # 对文件进行按行去重
        deduplicate_csv(path)

    except Exception as e:
        print(e)
        print("处理失败，文件路径为：{}".format(path))


def get_basic_info_from_tushare():
    # 初始化 tushare
    ts.set_token('30e6c0329269ab3e3ac6dfcc8737b274084e683ea121395597940bcc')  # 替换为您的 tushare token
    pro = ts.pro_api()

    # 查询当前所有正常上市交易的股票列表
    data_ts = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

    return data_ts

# if __name__ == "__main__":
#     code = '000001.SH'
#     data = xtdata.get_local_data(field_list=[], stock_code=[code], period='1d', start_time="20070101")
#     reformat_data_one_stock(data, code)


