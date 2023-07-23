# https://github.com/robertmartin8/MachineLearningStocks
# 代码通过训练一个随机森林，去寻找那些表现超过基准（标普五百）一定百分比的股票并投资。
# TODO: 需要适配自己的数据进行进一步修改
# https://easyai.tech/ai-definition/random-forest/
# “随机森林已经被证明在某些噪音较大的分类或回归问题上会过拟合。”，这需要我们充分的回测。
# “它可以出来很高维度（特征很多）的数据，并且不用降维，无需做特征选择。”
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from utils import data_string_to_float, status_calc


# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10


# def data_string_to_float(number_string):
#     """
#     The result of our regex search is a number stored as a string, but we need a float.
#         - Some of these strings say things like '25M' instead of 25000000.
#         - Some have 'N/A' in them.
#         - Some are negative (have '-' in front of the numbers).
#         - As an artifact of our regex, some values which were meant to be zero are instead '>0'.
#     We must process all of these cases accordingly.
#     :param number_string: the string output of our regex, which needs to be converted to a float.
#     :return: a float representation of the string, taking into account minus sign, unit, etc.
#     """
#     # Deal with zeroes and the sign
#     if ("N/A" in number_string) or ("NaN" in number_string):
#         return "N/A"
#     elif number_string == ">0":
#         return 0
#     elif "B" in number_string:
#         return float(number_string.replace("B", "")) * 1000000000
#     elif "M" in number_string:
#         return float(number_string.replace("M", "")) * 1000000
#     elif "K" in number_string:
#         return float(number_string.replace("K", "")) * 1000
#     else:
#         return float(number_string)


def status_calc(stock, sp500, outperformance=10):
    """A simple function to classify whether a stock outperformed the S&P500
    :param stock: stock price
    :param sp500: S&P500 price
    :param outperformance: stock is classified 1 if stock price > S&P500 price + outperformance
    :return: true/false
    """
    if outperformance < 0:
        raise ValueError("outperformance must be positive")
    return stock - sp500 >= outperformance


def build_data_set():
    """
    Reads the keystats.csv file and prepares it for scikit-learn
    :return: X_train and y_train numpy arrays
    """
    training_data = pd.read_csv("keystats.csv", index_col="Date")
    training_data.dropna(axis=0, how="any", inplace=True)
    features = training_data.columns[6:]

    X_train = training_data[features].values
    # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
    y_train = list(
        status_calc(
            training_data["stock_p_change"],
            training_data["SP500_p_change"],
            OUTPERFORMANCE,
        )
    )

    return X_train, y_train


def predict_stocks():
    X_train, y_train = build_data_set()
    # Remove the random_state parameter to generate actual predictions
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    # Now we get the actual data from which we want to generate predictions.
    data = pd.read_csv("forward_sample.csv", index_col="Date")
    data.dropna(axis=0, how="any", inplace=True)
    features = data.columns[6:]
    X_test = data[features].values
    z = data["Ticker"].values

    # Get the predicted tickers
    y_pred = clf.predict(X_test)
    if sum(y_pred) == 0:
        print("No stocks predicted!")
    else:
        invest_list = z[y_pred].tolist()
        print(
            f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {OUTPERFORMANCE}%:"
        )
        print(" ".join(invest_list))
        return invest_list


if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    predict_stocks()