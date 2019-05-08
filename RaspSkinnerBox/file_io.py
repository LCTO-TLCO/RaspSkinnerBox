#! /usr/bin python3
# coding:utf-8
from datetime import datetime
import logging
import csv
import os

logging.basicConfig(
    level=logging.DEBUG,
    filename='error.txt',
    filemode='a')

# define
logfile_path = 'epsilon-greedy.txt'
dispence_logfile_path = 'dispence_feed.csv'
nosepoke_logfile_path = 'nosepoke_log.csv'


class pycolor:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


colors = {'time over': pycolor.YELLOW,
          'reward': pycolor.GREEN,
          'failure': pycolor.YELLOW,
          'task called': pycolor.WHITE,
          }
integers = {'3': pycolor.GREEN,
            '5': pycolor.RED,
            '7': pycolor.YELLOW}


def set_dir():
    # os.chdir("../")
    None

def export(task_no: str, session_no: int, times: int, event_type: str, hole_no=0):
    logstring = ','.join([str(datetime.now()), task_no, str(session_no), str(times), event_type])
    with open(os.path.join("log", logfile_path), 'a+') as logfile:
        logfile.write(",".join([logstring, str(hole_no)]) + "\n")
        logfile.flush()
    print(add_color(logstring, str(hole_no)))


def add_color(string: str, integer: str):
    global colors, integers
    for keyword, new_color in colors.items():
        string = string.replace(keyword, "".join([new_color, keyword, pycolor.END]))
    for keyvalue, new_color in integers.items():
        integer = integer.replace(keyvalue, "".join([new_color, keyvalue, pycolor.END]))
    return ",".join([string, integer])


def magagine_log(reason, amount=1):
    """ 餌やりのログ 記入事項：「日付, 量(粒),報酬・精算」 """
    string = ','.join([str(datetime.now()), str(amount), reason])
    with open(os.path.join('log', 'dispence_feed.csv'), 'a+') as dispence_log_file:
        dispence_log_file.write(string + "\n")
        dispence_log_file.flush()


def select_preview_payoff():
    # read csv
    with open(os.path.join("log", 'dispence_feed.csv'), 'a+') as dispense_log_file:
        data = csv.reader(dispense_log_file)
        # select cols
        preview_payoff_time = -1
        if datetime.now().hour < 10:
            preview_payoff_time = datetime(datetime.today().year, datetime.today().month, datetime.today().day - 1,
                                           10, 0, 0)
        else:
            preview_payoff_time = datetime(datetime.today().year, datetime.today().month, datetime.today().day,
                                           10, 0, 0)


def last_session_id():
    if not os.path.exists(logfile_path):
        return 0
    else:
        with open(os.path.join("log", logfile_path), 'r') as logfile:
            last = logfile.readlines()[-1]
            return int(last[3]) + 1


def error_log(error):
    logging.exception(error)
    raise


def callback_rising(channel):
    all_nosepoke_log(channel, "RISING")


def callback_falling(channel):
    all_nosepoke_log(channel, "FALLING")


def all_nosepoke_log(channel: int, event_type: str):
    string = ','.join([str(datetime.now()), event_type, str(channel)])
    with open(os.path.join("log", nosepoke_logfile_path), 'a+') as poke_log:
        poke_log.write(string + "\n")
        poke_log.flush()


if __name__ == "__main__":
    export("test", 1, 1, "reward", "3/5")
    export("test", 1, 1, "time over", "3/5")
