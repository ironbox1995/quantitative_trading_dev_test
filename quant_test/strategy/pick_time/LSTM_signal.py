import numpy as np
import pandas as pd

from strategy.pick_time.deep_signal.model_setup import *


def create_predict_data(lst):
    # Create data from list
    increase_rate = [(lst[i] - lst[i-1])/lst[i-1] for i in range(1, len(lst))]
    data = np.array([increase_rate])
    # Normalize the data
    data = (data - np.mean(data)) / np.std(data)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    return data


def predict_new_data(lst):
    # https://bigquant.com/wiki/doc/dapan-too347vaWU 中写的择时方法显然要更好，单变量择时的话可能还是会有一些问题
    # 对于净值曲线，虽然只有单变量，但是也许可以基于大盘择时
    # Load the saved LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=1, output_size=1, dropout_rate=0, device=device).to(device)
    # 我觉得这里input_size是1其实有点奇怪，不过看起来就是这样的：https://developer.aliyun.com/article/1165191
    # model = LSTMAttentionModel(input_size=1, hidden_size=200, num_layers=1, output_size=1).to(device)
    model.load_state_dict(torch.load('model parameter file path here'))

    # Predict the target for new data
    new_data = create_predict_data(lst)
    prediction = model(torch.from_numpy(new_data).float().to(device))
    prediction = prediction.cpu().detach().numpy()[0][0]

    # if prediction > lst[-1] or pd.isnull(prediction):  
    if prediction > 0 or pd.isnull(prediction):  # 不一定非得大于0，我认为可以使任何一个大于0而又不太过分的数，这个数依赖于尝试。
        return 1
    else:
        return 0


def LSTM_signal(select_stock):

    length_prdt = 20
    select_stock["signal"] = 1
    for i in range(length_prdt, len(select_stock)):
        data_list = select_stock.iloc[i-length_prdt:i+1]['资金曲线'].tolist()
        signal_i = predict_new_data(data_list)
        select_stock.iloc[i, "signal"] = signal_i
    latest_signal = select_stock.tail(1)['signal'].iloc[0]
    select_stock['signal'] = select_stock['signal'].shift(1)
    select_stock['signal'].fillna(value=1, inplace=True)
    return select_stock, latest_signal
