import torch
import torch.nn as nn
from predictor.LSTM.data_loader import *
from predictor.LSTM.config import *


def load_model(model_path):
    """
    Load the saved model from the given path.
    """
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, data_loader):
    """
    Use the model to make predictions on the data provided by data_loader.
    """
    predictions = []
    with torch.no_grad():  # No need to track gradients
        for inputs, _ in data_loader:
            inputs = inputs.to(device)  # 将输入数据移至指定的设备
            outputs = model(inputs)
            predictions.extend(outputs.cpu().tolist())  # 将输出移回CPU，并存入列表
    return predictions


if __name__ == "__main__":
    data_filter = "小市值"

    # Load the model
    model_path = f'LSTM_reg_{predict_data_start_date}_{predict_data_end_date}-{period_type}-{data_filter}.pth'
    model = load_model(model_path)

    # Prepare your data loader for the data you want to predict
    prediction_loader = build_lstm_prediction_data_set(feature_li, predict_data_start_date, predict_data_end_date, period_type, 64, data_filter, sequence_length=20)

    # Make predictions
    predictions = predict(model, prediction_loader)

    # Do something with the predictions
    # print(predictions)

