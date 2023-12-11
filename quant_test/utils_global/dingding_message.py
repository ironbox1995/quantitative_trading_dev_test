# -*- coding: utf-8 -*-
import json
from datetime import datetime
import requests

from Config.global_config import *


# 函数：发送钉钉消息
def send_dingding(message, robot_id=dingding_robot_id, max_try_count=5):
    """
    出错会自动重发发送钉钉消息
    :param message: 你要发送的消息内容
    :param robot_id: 你的钉钉机器人ID
    :param max_try_count: 最多重试的次数
    """
    try_count = 0
    while True:
        try_count += 1
        try:
            msg = {
                "msgtype": "text",
                "text": {"content": message + '\n' + datetime.now().strftime("%m-%d %H:%M:%S")}}
            headers = {"Content-Type": "application/json;charset=utf-8"}
            url = 'https://oapi.dingtalk.com/robot/send?access_token=' + robot_id
            body = json.dumps(msg)
            requests.post(url, data=body, headers=headers)
            print('钉钉已发送')
            break
        except Exception as e:
            if try_count > max_try_count:
                print("发送钉钉失败：", e)
                break
            else:
                print("发送钉钉报错，重试：", e)


if __name__ == "__main__":
    send_dingding("交易测试信息。")  # 信息中必须有关键词“交易”，另外开着VPN发不了。
