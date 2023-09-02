import datetime

from trade.logger.logger_config import log
from utils_global.dingding_message import send_dingding


def record_log(msg, log_type='info', send=False):
    """
    记录日志
    :param msg: 日志信息
    :param log_type: 日志类型
    :return:
    """
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%H:%M:%S")
    log_msg = time_str + ' --> ' + msg
    print(log_msg)
    if log_type == 'info':
        log.info(msg=log_msg)
        if send:
            send_dingding(log_msg)
