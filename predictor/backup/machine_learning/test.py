from sklearn.metrics import precision_score, f1_score, mean_squared_error

from predictor.backup.machine_learning.data_loader import *
from predictor.backup.machine_learning.nn_model import *
from predictor.backup.machine_learning.model_config import *
from predictor.backup.machine_learning.ml_utils import *


def ML_model_tester(model_path, start_date, end_date, data_type):

    # Load from file
    if "regress" in model_path:
        X_train, y_train, X_test, y_test = build_regression_data_set(start_date, end_date, data_type)
    else:
        X_train, y_train, X_test, y_test = build_classify_data_set(start_date, end_date, data_type)

    with open(model_path, 'rb') as file:
        pickle_model = pickle.load(file)

    # Calculate the precision score and predict target values
    y_pred = pickle_model.predict(X_test)

    if "regress" in model_path:
        mse = mean_squared_error(y_test, y_pred)
        corr_coef = np.corrcoef(y_test, y_pred)[0][1]
        print("Correlation Coefficient: {}".format(corr_coef))
        print("Test MSE: {}".format(mse))
    else:
        precision = precision_score(X_test, y_pred)
        f1 = f1_score(X_test, y_pred)
        print("Test precision: {0:.2f} %".format(100 * precision))
        print("F1 score: {0:.2f} %".format(100 * f1))

    return y_pred


def DL_regress_model_tester(start_date, end_date, data_type, X_test, y_test,
                         hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer):

    # X_train, y_train, X_test, y_test = build_regression_data_set(start_date, end_date, data_type)
    output_size = 1
    model_path = "{}\strategy\pick_stock\machine_learning\model\FCN_regress_model_{}_{}-{}.pt".format(project_path, data_type, start_date, end_date)

    input_size = X_test.shape[1]
    model = FCN(input_size, hidden_size, output_size, num_hidden_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 不启用bn和dropout
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float()
        outputs = model(X_test_tensor)
        y_pred = outputs.numpy()
        mse = mean_squared_error(y_test, y_pred)
        corr_coef = np.corrcoef(y_test, y_pred)[0][1]
        print("Correlation Coefficient: {}".format(corr_coef))
        print("Test MSE: {}".format(mse))
        print("square root of the loss: {}".format(mse ** 0.5))

    return y_test, y_pred


def DL_classify_model_tester(start_date, end_date, data_type, X_test, y_test,
                          hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer):

    X_train, y_train, X_test, y_test = build_classify_data_set(start_date, end_date, data_type)
    output_size = len(np.unique(y_test))
    model_path = r"{}\strategy\pick_stock\machine_learning\model\FCN_classify_model_{}_{}-{}.pt".format(project_path, data_type, start_date, end_date)

    input_size = X_test.shape[1]
    model = FCN(input_size, hidden_size, output_size, num_hidden_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 不启用bn和dropout
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float()
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("actually positive rate: {}".format(sum(y_test) / len(y_test)))
        print("Test precision: {} ".format(precision))
        print("F1 score: {}".format(f1))

    return y_test, y_pred

