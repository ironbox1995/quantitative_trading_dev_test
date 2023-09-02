# coding=gbk

from get_strategy_function import get_strategy_function
from backtest.repick_time import *
from back_test_config import *
from utils_global.global_config import *
import warnings
from utils_global.dingding_message import *

warnings.filterwarnings('ignore')


def back_test_latest_result(strategy_name, select_stock_num, period_type, alpha, pick_time_mtd=""):
    pick_stock_strategy = get_strategy_function(strategy_name)

    if not Second_Board_available:
        strategy_name += "�޴�ҵ"
    if not STAR_Market_available:
        strategy_name += "�޿ƴ�"

    print('��������:', strategy_name)
    print('����:', period_type)
    print('��ʱ����:', pick_time_mtd)

    # ��������
    c_rate = 1 / 10000  # ������ ������֮ǰ��ͬ
    t_rate = 1 / 2000  # ӡ��˰

    # ===��������
    # ��pickle�ļ��ж�ȡ����õ����й�Ʊ����
    df = pd.read_pickle(
        r'{}\data\historical\processed_data\all_stock_data_{}.pkl'.format(project_path, period_type))
    # ===ɾ���¸������ղ����ס�������ͣ�Ĺ�Ʊ����Ϊ��Щ��Ʊ���¸������տ���ʱ�������롣
    df = df[df['����_�Ƿ���'] == 1]
    df = df[df['����_������ͣ'] == False]
    df = df[df['����_�Ƿ�ST'] == False]
    df = df[df['����_�Ƿ�����'] == False]

    # ѡ�ɲ��ԣ���ͨ�����벻ͬ�Ĳ��Խ����滻��
    session_id, df = pick_stock_strategy(df, select_stock_num)

    # ===����ѡ�й�Ʊ����
    # ��ѡ��ѡ�й�Ʊ
    df['��Ʊ����'] += ' '
    df['��Ʊ����'] += ' '
    group = df.groupby('��������')
    select_stock = pd.DataFrame()
    select_stock['��Ʊ����'] = group['��Ʊ����'].size()
    select_stock['�����Ʊ����'] = group['��Ʊ����'].sum()
    select_stock['�����Ʊ����'] = group['��Ʊ����'].sum()
    # ����select_stock�����һ��
    latest_selection = select_stock.tail(1)

    # ȥ�����һ�к��ٴ���һ��
    df.dropna(subset=['������ÿ���ǵ���'], inplace=True)
    group = df.groupby('��������')
    select_stock = pd.DataFrame()
    select_stock['��Ʊ����'] = group['��Ʊ����'].size()
    select_stock['�����Ʊ����'] = group['��Ʊ����'].sum()
    select_stock['�����Ʊ����'] = group['��Ʊ����'].sum()
    # ����������ÿ����ʽ����ߣ��Լ�ֻ��Ʊȡƽ��
    select_stock['ѡ��������ÿ���ʽ�����'] = group['������ÿ���ǵ���'].apply(
        lambda x: np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))
    # print(np.cumprod(np.array(list(x)) + 1, axis=1).mean(axis=0))  # ���ˣ������ʽ����ߣ��Լ�ֻ��Ʊȡƽ��

    # �۳�����������
    select_stock['ѡ��������ÿ���ʽ�����'] = select_stock['ѡ��������ÿ���ʽ�����'] * (1 - c_rate)  # �����в���׼�ĵط�
    # �۳����������ѡ�ӡ��˰�����һ����ʽ�����ֵ���۳�ӡ��˰��������
    select_stock['ѡ��������ÿ���ʽ�����'] = select_stock['ѡ��������ÿ���ʽ�����'].apply(
        lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate - t_rate)])

    # ���������������ǵ���
    select_stock['ѡ���������ǵ���'] = select_stock['ѡ��������ÿ���ʽ�����'].apply(lambda x: x[-1] - 1)
    # ����������ÿ����ǵ���
    select_stock['ѡ��������ÿ���ǵ���'] = select_stock['ѡ��������ÿ���ʽ�����'].apply(
        lambda x: list(pd.DataFrame([1] + x).pct_change()[0].iloc[1:]))
    del select_stock['ѡ��������ÿ���ʽ�����']

    # ���������ʽ�����
    select_stock.reset_index(inplace=True)
    select_stock['�ʽ�����'] = (select_stock['ѡ���������ǵ���'] + 1).cumprod()
    select_stock.set_index('��������', inplace=True)

    # �����ʽ�������ʱ
    if pick_time_mtd == "" or pick_time_mtd == "����ʱ":
        pick_time_mtd = "����ʱ"
        latest_selection['������ʱ�ź�'] = 1  # ��ʹ����ʱҲҪ���һ���źţ�������ȷ��ȡ��
    else:
        select_stock, latest_signal = pick_time(select_stock, pick_time_mtd)
        latest_selection['������ʱ�ź�'] = latest_signal

    # ����Q�е�ֵ
    select_stock['Q'] = select_stock['ѡ���������ǵ���'].copy()
    select_stock['Q'].iloc[1:] = 0
    for i in range(1, len(select_stock)):
        select_stock['Q'].iloc[i] = alpha * select_stock['ѡ���������ǵ���'].iloc[i] + (1 - alpha) * select_stock['Q'].iloc[i - 1]
    latest_Q = select_stock.tail(1)['Q'].iloc[0]
    latest_selection['Q'] = latest_Q
    select_stock['Q'] = select_stock['Q'].shift(1)
    select_stock['Q'].fillna(value=0, inplace=True)

    latest_selection.to_csv(
        r"{}\backtest\latest_selection\����ѡ��_{}_{}_ѡ{}_{}.csv"
            .format(project_path, strategy_name, period_type, select_stock_num, pick_time_mtd), encoding='gbk')

    return select_stock


if __name__ == "__main__":

    for strategy_name in strategy_li:
        for period_type in period_type_li:
            for select_stock_num in select_stock_num_li:
                pick_time_mtd = pick_time_mtd_dct[strategy_name]
                try:
                    back_test_latest_result(strategy_name, select_stock_num, period_type, ALPHA, pick_time_mtd)
                    back_test_latest_result(strategy_name, select_stock_num, period_type, ALPHA, "����ʱ")
                except Exception as e:
                    msg = "���ײ���������{}������ʧ�ܣ�period_type:{}, select_stock_num:{}".format(strategy_name, period_type,
                                                                                select_stock_num)
                    print(msg)
                    send_dingding(msg)
                    print(e)
    send_dingding("���ײ�����ִ�� ���½����� �ɹ���")
