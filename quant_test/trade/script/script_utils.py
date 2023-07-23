# -*- coding: utf-8 -*-
import subprocess
from datetime import datetime, timedelta
import chinese_calendar as calendar


def execute_script_in_virtualenv(script_path, venv_path=r'F:\env_often_used'):
    command = f"{venv_path}\\Scripts\\python.exe {script_path}"
    subprocess.run(command, shell=True)


def first_workday_in_period():
    today = datetime.today().date()
    previous_day = today - timedelta(days=1)
    next_day = today + timedelta(days=1)

    previous_day_not_workday = not calendar.is_workday(previous_day)
    today_is_workday = calendar.is_workday(today)
    next_day_is_workday = calendar.is_workday(next_day)

    return previous_day_not_workday and today_is_workday and next_day_is_workday


def last_workday_in_period():
    today = datetime.today().date()
    next_day = today + timedelta(days=1)

    today_is_workday = calendar.is_workday(today)
    next_day_not_workday = not calendar.is_workday(next_day)

    return today_is_workday and next_day_not_workday
