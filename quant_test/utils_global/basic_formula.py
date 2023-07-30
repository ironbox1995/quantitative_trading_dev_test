def CAGR_calculator(initial_value, final_value, num_of_years):
    """
    年复合增长率：Compound Annual Growth Rate，简称CAGR
    :param initial_value:初始值
    :param final_value:最终值
    :param num_of_years:年数
    :return:CAGR
    """
    return pow(final_value/initial_value, 1/num_of_years) - 1


def compound_interest_calculator(initial_value, annual_interest_rate, number_of_years):
    """
    复利计算器
    :param initial_value: 初始值
    :param annual_interest_rate: 年利率
    :param number_of_years: 年数
    :return: 最终值
    """
    return initial_value * pow(1+annual_interest_rate, number_of_years)


def increase_rate(initial_value, final_value):
    return (final_value - initial_value)/initial_value


if __name__ == "__main__":
    print(CAGR_calculator(1, 1000, 20))