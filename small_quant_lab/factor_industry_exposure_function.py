# https://bbs.quantclass.cn/thread/26760

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# warnings.filterwarnings('ignore')

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 10000)  # 最多显示数据的行数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 配置matplotlib显示中文
plt.rcParams['axes.unicode_minus'] = False  # 修复负号问题
plt.rcParams["figure.dpi"] = 300  # 配置分辨率，为了论坛能够显示更清晰的PNG图片


def curve_ind_factor_boxplot(df: pd.DataFrame, factor: str, industry: str, start_date: str = '20100101',
                             end_date: str = '20230701') -> None:
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df['因子排名'] = df.groupby('交易日期')[factor].rank(ascending=False)  # 小市值为False，不同因子取值不同
    df['涨跌幅排名'] = df.groupby('交易日期')['下周期涨跌幅'].rank(ascending=True)

    ind = df[industry]
    ind_df = pd.get_dummies(ind, columns=[industry], dummy_na=False, drop_first=False)
    ind_col = list(ind_df.columns)
    df = pd.concat([df, ind_df], axis=1)

    df_list = []
    # results = pd.DataFrame()
    for _ in tqdm(ind_col):
        _factor = df[df[_] == True]['因子排名'] * (df['涨跌幅排名'])  # 选择出df中满足某个条件（df[_] == True）的行，然后再从这些行中选取出索引为factor的列
        _factor.reset_index(inplace=True, drop=True)
        df_list.append(_factor)

    factor_df = pd.concat(df_list, axis=1, keys=ind_col)
    factor_df.reset_index(inplace=True, drop=True)

    plt.figure(figsize=(10, 5))
    factor_df.boxplot(ind_col, showfliers=False)
    plt.xticks(rotation=45)
    plt.title(f"处理后{factor}因子行业分布")
    plt.savefig(f'{factor}因子行业分布.png')
    print(f"{factor}因子行业分布图导出完毕!")
    plt.show()
    plt.close()

    # plt.figure(figsize=(20, 10))
    # plt.bar(results.index, results['相关性'])
    # # results.plot(kind='bar')
    # plt.xticks(rotation=45)
    # plt.title(f"处理后{factor}因子行业收益相关性分布")
    # plt.savefig(f'{factor}因子行业收益相关性分布.png')
    # print(f"{factor}因子行业收益相关性分布图导出完毕!")
    # plt.show()
    # plt.close()


# 因子市值分布箱型图
def curve_cap_factor_boxplot(df: pd.DataFrame, factor: str, start_date: str = '20100101',
                             end_date: str = '20230701'):
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]

    spilt_labels = ['超小型', '小型', '中型', '大型', '超大型']
    df['市值划分'] = pd.qcut(df['总市值 （万元）'], q=5, labels=spilt_labels)
    df.sort_values(by=['交易日期', '总市值 （万元）', '市值划分'], inplace=True)

    cap_list = []
    for _ in tqdm(spilt_labels):
        _factor = df[df['市值划分'] == _][factor] * (df['涨跌幅排名'])
        _factor.reset_index(inplace=True, drop=True)
        cap_list.append(_factor)

    factor_df = pd.concat(cap_list, axis=1, keys=spilt_labels)
    factor_df.reset_index(inplace=True, drop=True)

    plt.figure(figsize=(20, 10))
    factor_df.boxplot(spilt_labels, showfliers=False)
    plt.xticks(rotation=45)
    plt.title(f"处理后{factor}因子市值分布")
    plt.savefig(f'{factor}因子市值分布.png')
    print(f"{factor}因子行业市值图导出完毕!")
    plt.show()
    plt.close()


def rank_IC_calculator(df: pd.DataFrame, factor: str, industry: str, start_date: str = '20100101',
                             end_date: str = '20230701'):
    # TODO: 这个算出的结果过于离谱，显然是错了，可以参考郭毅老板的改改
    #  https://bbs.quantclass.cn/thread/13238
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]
    df['因子排名'] = df.groupby('交易日期')[factor].rank(ascending=True)
    df['涨跌幅排名'] = df.groupby('交易日期')['下周期涨跌幅'].rank(ascending=True)

    ind = df[industry]
    ind_df = pd.get_dummies(ind, columns=[industry], dummy_na=False, drop_first=False)
    ind_col = list(ind_df.columns)
    df = pd.concat([df, ind_df], axis=1)

    dct = {}
    for _ in tqdm(ind_col):
        ic_value = df[df[_] == True]['因子排名'].corr(df[df[_] == True]['涨跌幅排名'])
        # print("因子：{}对行业：{}的IC值为：{}".format(factor, _, ic_value))
        dct[_] = ic_value

    return dct


def normal_IC_calculator(df: pd.DataFrame, factor: str, industry: str, start_date: str = '20100101',
                             end_date: str = '20230701'):
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]

    ind = df[industry]
    ind_df = pd.get_dummies(ind, columns=[industry], dummy_na=False, drop_first=False)
    ind_col = list(ind_df.columns)
    df = pd.concat([df, ind_df], axis=1)

    dct = {}
    for _ in tqdm(ind_col):
        ic_value = df[df[_] == True][factor].corr(df[df[_] == True]['下周期涨跌幅'])
        # print("因子：{}对行业：{}的IC值为：{}".format(factor, _, ic_value))
        dct[_] = ic_value

    return dct


# 因子市值分布箱型图
# qcut:https://blog.csdn.net/yeshang_lady/article/details/107957020
def curve_cap_factor_boxplot1(df: pd.DataFrame, factor: str, start_date: str = '20100101',
                             end_date: str = '20230701'):
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df = df[(df['交易日期'] >= pd.to_datetime(start_date)) & (df['交易日期'] <= pd.to_datetime(end_date))]

    spilt_labels = [i for i in range(20)]
    df['因子划分'] = pd.qcut(df[factor], q=len(spilt_labels), labels=spilt_labels, duplicates='drop')
    df.sort_values(by=['交易日期', factor, '因子划分'], inplace=True)

    cap_list = []
    for _ in tqdm(spilt_labels):
        _factor = df[df['因子划分'] == _]['下周期涨跌幅'].mean()
        # _factor.reset_index(inplace=True, drop=True)
        cap_list.append(_factor)

    # factor_df = pd.concat(cap_list, axis=1, keys=spilt_labels)
    # factor_df.reset_index(inplace=True, drop=True)

    # plt.figure(figsize=(20, 10))
    # 绘制线形图
    plt.bar(spilt_labels, cap_list)
    # factor_df.boxplot(spilt_labels, showfliers=False)
    # plt.xticks(rotation=45)
    plt.title(f"处理后{factor}因子分箱分布")
    plt.savefig(f'pic/{factor}因子分箱分布{start_date}-{end_date}.png')
    print(f"{factor}因子分箱图导出完毕!")
    # plt.show()
    plt.clf()
    plt.close()