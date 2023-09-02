# -*- coding: utf-8 -*-
from trade.script.script_utils import *
from utils_global.global_config import *


# 周一早上执行这个
if __name__ == "__main__":
    file_path = r"{}\trade\buy_main.py".format(project_path)
    execute_script_in_virtualenv(file_path)
