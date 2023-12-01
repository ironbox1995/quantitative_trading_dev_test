import torch
import torch.nn as nn
from data_loader import *
from config import *


def load_model(model_path):
    """
    Load the saved model from the given path.
    """
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model


def predict(model, data_loader):
    """
    Use the model to make predictions on the data provided by data_loader.
    """
    predictions = []
    with torch.no_grad():  # No need to track gradients
        for inputs, _ in data_loader:
            outputs = model(inputs)
            predictions.extend(outputs.tolist())  # Convert to list and store
    return predictions


if __name__ == "__main__":
    # Load the model
    model_path = f'FCN_reg_{predict_data_start_date}_{predict_data_end_date}-{period_type}.pth'
    model = load_model(model_path)

    # Prepare your data loader for the data you want to predict
    prediction_loader = build_prediction_data_set(feature_li, predict_data_start_date, predict_data_end_date, period_type, 64)

    # Make predictions
    predictions = predict(model, prediction_loader)

    # Do something with the predictions
    # print(predictions)

