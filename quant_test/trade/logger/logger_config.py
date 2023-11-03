import datetime
import logging as log
from Config.global_config import *

# ========== 初始化 ==========
root_path = project_path
# region 发送日志相关
log_path = root_path + '/logs/'
log.basicConfig(filename=log_path + '%s_日志.log' % datetime.datetime.now().strftime('%Y-%m-%d'), level=log.INFO)