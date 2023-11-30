import pickle
import datetime
import torch
import numpy as np


def load_scaler(model_path, data_type, train_start_date, train_end_date):

    if "random_forest" in model_path:
        if "regress" in model_path:
            with open('model_setup/random_forest_regress_model-scaler-{}_{}-{}.pkl'.format(data_type, train_start_date, train_end_date), 'rb') as file:
                loaded_scaler = pickle.load(file)
        else:
            with open('model_setup/random_forest_classify_model-scaler-{}_{}-{}.pkl'.format(data_type, train_start_date, train_end_date), 'rb') as file:
                loaded_scaler = pickle.load(file)
    elif "SV" in model_path:
        if "regress" in model_path:
            with open('model_setup/SVR_regress_model-scaler-{}_{}-{}.pkl'.format(data_type, train_start_date, train_end_date), 'rb') as file:
                loaded_scaler = pickle.load(file)
        else:
            with open('model_setup/SVC_classify_model-scaler-{}_{}-{}.pkl'.format(data_type, train_start_date, train_end_date), 'rb') as file:
                loaded_scaler = pickle.load(file)
    else:
        if "regress" in model_path:
            with open('model_setup/FCN_regress_model-scaler-{}_{}-{}.pkl'.format(data_type, train_start_date, train_end_date), 'rb') as file:
                loaded_scaler = pickle.load(file)
        else:
            with open('model_setup/FCN_classify_model-scaler-{}_{}-{}.pkl'.format(data_type, train_start_date, train_end_date), 'rb') as file:
                loaded_scaler = pickle.load(file)

    return loaded_scaler


def get_next_year_first_last_days(date_str):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    next_year = date.year + 1
    first_day_next_year = datetime.datetime(next_year, 1, 1).strftime('%Y-%m-%d')
    last_day_next_year = datetime.datetime(next_year, 12, 31).strftime('%Y-%m-%d')
    return first_day_next_year, last_day_next_year


def get_next_month_first_last_days(date_str):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    next_month = date.month + 1
    if next_month > 12:
        next_month = 1
        next_year = date.year + 1
    else:
        next_year = date.year
    first_day_next_month = datetime.datetime(next_year, next_month, 1).strftime('%Y-%m-%d')
    last_day_next_month = (datetime.datetime(next_year, next_month + 1, 1) - datetime.timedelta(days=1)).strftime(
        '%Y-%m-%d')
    return first_day_next_month, last_day_next_month


def subtract_months(date_str, months):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    month = date.month
    day = date.day
    for i in range(months):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    new_date = datetime.datetime(year, month, day).strftime('%Y-%m-%d')
    return new_date


def get_every_month_first_last_days(start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    current_month = start_date.month
    current_year = start_date.year
    first_day_current_month = start_date_str
    last_day_current_month = get_next_month_first_last_days(start_date_str)[1]
    result = []
    while current_year < end_date.year or (current_year == end_date.year and current_month <= end_date.month):
        result.append([datetime.datetime(current_year, current_month, 1),
                       datetime.datetime.strptime(last_day_current_month, '%Y-%m-%d')])
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        first_day_current_month = datetime.datetime(current_year, current_month, 1).strftime('%Y-%m-%d')
        last_day_current_month = get_next_month_first_last_days(first_day_current_month)[1]
    return result


def get_interval_first_last_day(start_date_str, end_date_str, months):
    result = get_every_month_first_last_days(start_date_str, end_date_str)
    for time_interval in result:
        time_interval[0] = subtract_months(time_interval[0], months)
    return result


# Define a function to evaluate the model_setup on the validation set
def evaluate_regress_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss


# Define a function to evaluate the model_setup on the validation set
def evaluate_classify_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    return val_loss/len(val_loader)


def correlation_coefficient(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator
