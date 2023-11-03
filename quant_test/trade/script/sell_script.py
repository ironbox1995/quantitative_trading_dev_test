# -*- coding: utf-8 -*-
from trade.script.script_utils import *
from Config.global_config import *


# 周五下午执行这个
if __name__ == "__main__":
    file_path = r"{}\trade\sell_main.py".format(project_path)
    execute_script_in_virtualenv(file_path)
