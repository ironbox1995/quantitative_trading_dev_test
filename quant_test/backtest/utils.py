import random
import string
import datetime


def generate_serial_number():
    # 取32位随机序列号
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))


def get_current_date():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return current_date


if __name__ == "__main__":
    serial_number = generate_serial_number()
    print(serial_number)
