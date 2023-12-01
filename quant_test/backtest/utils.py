import random
import string
import datetime
from Config.back_test_config import *


def generate_serial_number():
    # 取32位随机序列号
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))


def get_current_date():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return current_date


def cal_limit_threshold(code):
    # 对科创板进行特殊处理
    if code[:2] == '68' and code[-2:] == 'SH':
        limit_up_threshold = 0.2
        limit_down_threshold = -0.2
    elif code[:2] == '30' and code[-2:] == 'SZ':
        limit_up_threshold = 0.2
        limit_down_threshold = -0.2
    else:
        limit_up_threshold = 0.1
        limit_down_threshold = -0.1

    # 某一个不止损则设为100%
    if not limit_up_take_profit:
        limit_up_threshold = 1.0
    if not limit_down_stop_loss:
        limit_down_threshold = -1.0

    return limit_up_threshold, limit_down_threshold


def process_limit_up_and_down(daily_changes, next_day_open_sell_change, code):
    limit_up, limit_down = cal_limit_threshold(code)
    for i in range(len(daily_changes)):
        # Check for limit-up or limit-down
        if daily_changes[i] >= limit_up or daily_changes[i] <= limit_down:
            # Use 'next_day_open_sell_change' for the day after limit-up/down
            if i+1 < len(daily_changes):
                daily_changes[i+1] = next_day_open_sell_change
                # Set changes to 0 for the days after
                daily_changes[i+2:] = [0] * (len(daily_changes) - i - 2)
            break
    return daily_changes


if __name__ == "__main__":
    serial_number = generate_serial_number()
    print(serial_number)
