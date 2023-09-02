from strategy.pick_stock.machine_learning.data_loader import *
from strategy.pick_stock.machine_learning.nn_model import *
from strategy.pick_stock.machine_learning.model_config import *
from strategy.pick_stock.machine_learning.ml_utils import *


def ML_model_predictor(pick_stock_df, period_type, model_type):
    # 根据实际情况选择训练时间，每个模型只能用来预测一年，因此模型的数量必须等于回测的年份，也可以考虑增加模型更新频率，说不定会更好
    # model_time_pair_tpl = (('2007-01-01', '2008-12-31'), ('2008-01-01', '2009-12-31'))

    for time_pair in model_time_pair_tpl:
        time_pair_next_year = get_next_year_first_last_days(time_pair[1])  # 用上一年或几年的数据预测下一年的情况
        pick_stock_df_interval = pick_stock_df[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
                                      & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1]))]
        if len(pick_stock_df_interval) == 0:
            continue

        model_path = r'{}\strategy\pick_stock\machine_learning\model\{}_model_{}_{}-{}.pt'.format(project_path, model_type, period_type, time_pair[0], time_pair[1])

        with open(model_path, 'rb') as file:
            pickle_model = pickle.load(file)

        if "regress" in model_path:
            X_predict = build_regression_prediction_data_set(pick_stock_df_interval)
        else:
            X_predict = build_classify_prediction_data_set(pick_stock_df_interval)

        Y_predict = pickle_model.predict(X_predict)

        pick_stock_df.loc[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
                          & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1])), "机器学习预测值"] = Y_predict

    return pick_stock_df


# def DL_model_predictor(pick_stock_df, data_type, model_type, hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer):
#     # 根据实际情况选择训练时间，每个模型只能用来预测一年，因此模型的数量必须等于回测的年份，也可以考虑增加模型更新频率，说不定会更好
#     # model_time_pair_tpl = (('2007-01-01', '2008-12-31'), ('2008-01-01', '2009-12-31'))
#
#     for time_pair in model_time_pair_tpl:
#         time_pair_next_year = get_next_year_first_last_days(time_pair[1])  # 用上一年或几年的数据预测下一年的情况
#         pick_stock_df_interval = pick_stock_df[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
#                                       & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1]))]
#         if len(pick_stock_df_interval) == 0:
#             continue
#
#         if "regress" == model_type:
#             X_predict = build_regression_prediction_data_set(pick_stock_df_interval)
#             model_path = "F:\quantitative_trading_dev_test\quant_test\strategy\pick_stock\machine_learning\model\FCN_regress_model_{}_{}-{}.pt".format(
#                 data_type, time_pair[0], time_pair[1])
#             output_size = 1
#         else:
#             X_predict = build_classify_prediction_data_set(pick_stock_df_interval)
#             model_path = "F:\quantitative_trading_dev_test\quant_test\strategy\pick_stock\machine_learning\model\FCN_classify_model_{}_{}-{}.pt".format(
#                 data_type, time_pair[0], time_pair[1])
#             output_size = 2
#
#         input_size = X_predict.shape[1]
#         model = FCN(input_size, hidden_size, output_size, num_hidden_layers)
#         model.load_state_dict(torch.load(model_path))
#
#         X_predict_tensor = torch.FloatTensor(X_predict)
#         if "regress" == model_type:
#             y_pred = model(X_predict_tensor).detach().numpy()
#         else:
#             outputs = model(X_predict_tensor)
#             _, predicted = torch.max(outputs.data, 1)
#             y_pred = predicted.numpy()
#
#         pick_stock_df.loc[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
#                           & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1])), "机器学习预测值"] = y_pred
#
#     return pick_stock_df


def DL_model_regress_predictor(pick_stock_df, data_type, hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer):
    for time_pair in model_time_pair_tpl:
        time_pair_next_year = get_next_year_first_last_days(time_pair[1])
        pick_stock_df_interval = pick_stock_df[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
                                      & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1]))]
        if len(pick_stock_df_interval) == 0:
            continue

        X_predict = build_regression_prediction_data_set(pick_stock_df_interval)
        model_path = "{}\strategy\pick_stock\machine_learning\model\FCN_regress_model_{}_{}-{}.pt".format(project_path,
            data_type, time_pair[0], time_pair[1])
        output_size = 1

        input_size = X_predict.shape[1]
        model = FCN(input_size, hidden_size, output_size, num_hidden_layers)
        model.load_state_dict(torch.load(model_path))

        X_predict_tensor = torch.FloatTensor(X_predict)
        y_pred = model(X_predict_tensor).detach().numpy()

        pick_stock_df.loc[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
                          & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1])), "机器学习预测值"] = y_pred

    return pick_stock_df


def DL_model_classify_predictor(pick_stock_df, data_type, hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer):
    for time_pair in model_time_pair_tpl:
        time_pair_next_year = get_next_year_first_last_days(time_pair[1])
        pick_stock_df_interval = pick_stock_df[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
                                      & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1]))]
        if len(pick_stock_df_interval) == 0:
            continue

        X_predict = build_classify_prediction_data_set(pick_stock_df_interval)
        model_path = "{}\strategy\pick_stock\machine_learning\model\FCN_classify_model_{}_{}-{}.pt".format(project_path,
            data_type, time_pair[0], time_pair[1])
        output_size = 2

        input_size = X_predict.shape[1]
        model = FCN(input_size, hidden_size, output_size, num_hidden_layers)
        model.load_state_dict(torch.load(model_path))

        X_predict_tensor = torch.FloatTensor(X_predict)
        outputs = model(X_predict_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()

        pick_stock_df.loc[(pick_stock_df['交易日期'] >= pd.to_datetime(time_pair_next_year[0]))
                          & (pick_stock_df['交易日期'] <= pd.to_datetime(time_pair_next_year[1])), "机器学习预测值"] = y_pred

    return pick_stock_df
