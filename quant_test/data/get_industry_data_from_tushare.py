import pandas as pd
import tushare as ts
from processing.data_config import *
import time


def deduplicate_industry_data(df):
    # 读取 CSV 文件
    # df = pd.read_csv(r'{}\data\historical\tushare_industry_data\industry_data.csv'.format(project_path))

    df = df[df['is_new'] == "Y"]

    df['申万三级行业'] = df['index_code'].map(ind3_code_to_name_dct)
    df['申万二级行业'] = df['申万三级行业'].map(ind3_to_ind2_dct)
    df['申万一级行业'] = df['申万三级行业'].map(ind3_to_ind1_dct)

    # df.drop_duplicates(subset=['index_code', 'con_code'], keep='last', inplace=True)

    # 将结果写回 CSV 文件
    # df.to_csv(r'{}\data\historical\tushare_industry_data\industry_data_clean.csv'.format(project_path), index=False)
    return df


def get_tushare_industry_data_main():

    # 设置tushare的token，可以在tushare官网（https://tushare.pro/）申请免费token
    ts.set_token('30e6c0329269ab3e3ac6dfcc8737b274084e683ea121395597940bcc')

    # 初始化tushare pro接口
    pro = ts.pro_api()

    df_sw_l3 = pro.index_classify(level='L3', src='SW2021')
    sw_l3_li = df_sw_l3['index_code'].tolist()

    df_list = []
    for sw_l3 in sw_l3_li:
        print("正在获取{}行业数据。".format(sw_l3))
        df = pro.index_member(index_code=sw_l3)
        df_list.append(df)
        time.sleep(0.5)
        print("{}行业数据获取完成。".format(sw_l3))
    all_industry_data = pd.concat(df_list, ignore_index=True)
    all_industry_data = deduplicate_industry_data(all_industry_data)
    all_industry_data.to_csv(r'{}\data\historical\tushare_industry_data\industry_data.csv'.format(project_path))


if __name__ == "__main__":
    get_tushare_industry_data_main()
