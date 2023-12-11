from data.processing.reformat_utils import *
from Config.data_config import *
from Config.global_config import *
import time


# def get_financial_indicator_data_from_tushare(pro, stock_code, start_date, end_date):
#     df_lst = []
#     last_date = start_date
#     while int(last_date) < int(end_date):
#         # 获取财务指标数据，每次最多返回60条
#         df = pro.query('fina_indicator', ts_code=stock_code, start_date=start_date)
#         print(df)
#         if len(df) < 1:
#             last_date = "99999999"
#         else:
#             print(df.iloc[0]["ann_date"])
#             last_date = df.iloc[0]["ann_date"]
#             df_lst.append(df)
#     financial_indicator_df = pd.concat(df_lst, axis=0, ignore_index=True)
#     financial_indicator_df = financial_indicator_df.drop_duplicates(subset=['end_date'], keep='last')
#     return financial_indicator_df


def get_financial_data_from_tushare(pro, stock_code, start_date, end_date):
    # 获取利润表数据
    df_income = pro.income(ts_code=stock_code, start_date=start_date, end_date=end_date)
    if '其他收益' not in df_income.columns:  # tushare数据此列往往为空
        df_income['其他收益'] = 0
    # 资产负债表数据
    df_balance = pro.balancesheet(ts_code=stock_code, start_date=start_date, end_date=end_date)
    # 现金流数据
    df_cash_flow = pro.cashflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
    # 财务指标数据
    # df_fin_ind = get_financial_indicator_data_from_tushare(pro, stock_code, start_date, end_date)
    df_fin_ind = pro.query('fina_indicator', ts_code=stock_code, start_date=start_date, end_date=end_date)

    finance_df = pd.merge(pd.merge(df_balance, df_income, on=['ts_code', 'ann_date', 'f_ann_date', 'end_date'])
                          , df_cash_flow, on=['ts_code', 'ann_date', 'f_ann_date', 'end_date'])
    finance_df = pd.merge(finance_df, df_fin_ind, on=['ts_code', 'ann_date', 'end_date'])

    fin_rename_dct = {**tushare_balance_columns_name_dct, **tushare_income_columns_name_dct,
                      **tushare_cashflow_columns_name_dct, **tushare_fina_indicator_columns_name_dct}
    finance_df.rename(columns=fin_rename_dct, inplace=True)
    # print(finance_df)
    # finance_df.to_csv("列名.csv", encoding='gbk')
    finance_data_column_li = ['股票代码', '公告日期', '报告期', '短期借款', '长期借款', '应付债券', '一年内到期的非流动负债', '营业总收入', '应付利息', '应付手续费及佣金', '减:销售费用', '减:管理费用',
         '研发费用', '减:资产减值损失', '固定资产折旧、油气资产折耗、生产性生物资产折旧', '无形资产摊销', '长期待摊费用摊销', '其他收益', '减:营业税金及附加', '减:营业成本',
         '净利润(不含少数股东损益)', '归属于母公司(或股东)的综合收益总额', '货币资金', '流动负债合计', '非流动负债合计', '经营活动产生的现金流量净额', '净利润', '营业总成本']
    finance_df = finance_df[finance_data_column_li + fin_ind_col_final]
    # 重命名与邢大的代码中不一样的部分列
    col_rename_dct = {'公告日期': '披露日期', '报告期': '截止日期', '应付利息': '负债应付利息', '减:营业成本': '营业成本', '减:管理费用': '管理费用',
                      '减:销售费用': '销售费用',
                      '固定资产折旧、油气资产折耗、生产性生物资产折旧': '固定资产折旧、油气资产折耗、生产性物资折旧', '其他收益': '其他综合利益',
                      '净利润(不含少数股东损益)': '归母净利润', '归属于母公司(或股东)的综合收益总额': '归母所有者权益合计',
                      '净利润(收入表)': '净利润', '减:资产减值损失': '资产减值损失', '减:营业税金及附加': '税金及附加'}
    finance_df.rename(columns=col_rename_dct, inplace=True)
    finance_df.sort_values(by=['披露日期'], ascending=True, inplace=True)
    finance_df.fillna(method='ffill', inplace=True)  # 缺失数据根据前值补充
    finance_df.fillna(0, inplace=True)  # 仍然缺失的财务数据补0

    return finance_df


def update_and_save_financial_data(pro):
    """
    获取并保存数据
    :param pro: 接口
    :return:
    """
    # 仅返回上市的
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,market,industry,'
                                                                         'list_date')

    print("开始从tushare下载历史财务数据，本次开始下载的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    for i in range(len(data)):
        try:
            stock_code = data.iloc[i]['ts_code']
            if 'BJ' in stock_code:
                print("不下载北京证券交易所股票数据：{}".format(stock_code))
                continue
            # 构建存储文件路径
            path = r'{}\data\historical\tushare_financial_data\{}.csv'.format(project_path, stock_code)
            # 计算起止日期
            today = get_today()
            if os.path.exists(path):
                first_day_of_year = datetime.date(datetime.date.today().year, 1, 1)
                start_date = first_day_of_year.strftime("%Y%m%d")  # 将日期格式化为字符串，格式为"YYYYMMDD"
            else:
                start_date = '20070101'
            one_financial_data_tushare = get_financial_data_from_tushare(pro, stock_code, start_date=start_date,
                                                                         end_date=today)

            # 保存数据
            save_financial_data(path, "财务", stock_code, one_financial_data_tushare)
            time.sleep(0.2)
        except Exception as e:
            stock_code = data.iloc[i]['ts_code']
            print(stock_code + "数据获取失败：", e)

    print("从tushare下载历史财务数据结束，本次下载结束的时间为：", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


def get_tushare_financial_data_main():
    """
    每次下载一年的数据，后续有去重方法
    :return:
    """

    # 设置tushare的token，可以在tushare官网（https://tushare.pro/）申请免费token
    ts.set_token(tushare_token)

    # 初始化tushare pro接口
    pro = ts.pro_api()

    update_and_save_financial_data(pro)


if __name__ == "__main__":
    get_tushare_financial_data_main()
