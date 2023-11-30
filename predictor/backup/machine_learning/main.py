from machine_learning.train import *
from machine_learning.test import *


def main(method="regression"):
    # TODO：将所有数据一股脑喂给模型或许是不可行的，应该通过行业或者其他方式筛选数据。
    data_type = "M"

    for time_pair in model_time_pair_tpl:
        if method == "regression":
            X_train, y_train, X_test, y_test = build_regression_data_set(time_pair[0], time_pair[1], data_type)
            train_FCN_regress_model(time_pair[0], time_pair[1], data_type, X_train, y_train, fcn_patience, fcn_hidden_size,
                                    fcn_hidden_layer, fcn_batch_size)
            y_test, y_pred = DL_regress_model_tester(time_pair[0], time_pair[1], data_type, X_test, y_test,
                                                     hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer)

        else:
            X_train, y_train, X_test, y_test = build_classify_data_set(time_pair[0], time_pair[1], data_type)
            train_FCN_classify_model(time_pair[0], time_pair[1], data_type, X_train, y_train, fcn_patience, fcn_hidden_size,
                                     fcn_hidden_layer, fcn_patience, fcn_batch_size)
            y_test, y_pred = DL_classify_model_tester(time_pair[0], time_pair[1], data_type, X_test, y_test,
                                                     hidden_size=fcn_hidden_size, num_hidden_layers=fcn_hidden_layer)

        plt.scatter(y_test, y_pred, s=1)
        plt.xlabel('y_test')
        plt.ylabel('y_pred')
        plt.show()


if __name__ == "__main__":
    main("regression")
