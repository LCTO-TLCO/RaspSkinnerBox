#! /usr/bin python3
# coding:utf-8
import shutil
from datetime import datetime, timedelta
import logging
import csv
import os
from collections import OrderedDict
import json
from defines import setting_file
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    filename='error.txt',
    filemode='a',
    format='%(asctime)s %(levelname)-8s %(module)-18s %(funcName)-10s %(lineno)4s: %(message)s')

# define
logfile_path = 'no{}_action.csv'
dispence_logfile_path = 'no{}_dispencer.csv'
nosepoke_logfile_path = 'no{}_nosepoke.csv'
settings_logfile_path = 'no{}_task_settings.json'
daily_logfile_path = 'no{}_daily_feed.csv'
ex_flow = OrderedDict({})
# ex_flow = OrderedDict()
ex_flow.update(json.load(open(setting_file, "r"), object_pairs_hook=OrderedDict))


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


colors = {'time over': pycolor.PURPLE,
          'reward': pycolor.GREEN,
          'failure': pycolor.YELLOW,
          'task called': pycolor.WHITE,
          }
integers = {'1': pycolor.CYAN,
            '3': pycolor.GREEN,
            '5': pycolor.RED,
            '7': pycolor.YELLOW,
            '9': pycolor.BLUE}


def set_dir():
    pass
    # if (os.path.basename(os.getcwd()) == "log"):
    #     return
    # os.chdir("./log")


def export(task_no: str, session_no: int, times: int, event_type: str, hole_no=0):
    logstring = ','.join([str(datetime.now()), task_no, str(session_no), str(times), event_type])
    with open(os.path.join("log", logfile_path), 'a+') as logfile:
        head = ["task", "session", "rewardnum", "action", "cond"]
        if os.path.getsize(logfile_path):
            logfile.write(",".join(head) + "\n")
        logfile.write(",".join([logstring, str(hole_no)]) + "\n")
        logfile.flush()
    print(add_color(logstring, str(hole_no)))


def add_color(string: str, integer: str):
    global colors, integers
    for keyword, new_color in colors.items():
        string = string.replace(keyword, "".join([new_color, keyword, pycolor.END]))
    retval = ""
    for st in integer:
        if st in integers.keys():
            retval += st.replace(st, "".join([integers[st], st, pycolor.END]))
        else:
            retval += st

    return ",".join([string, retval])


def magagine_log(reason, amount=1):
    """ 餌やりのログ 記入事項：「日付, 量(粒),報酬・精算」 """
    string = ','.join([str(datetime.now()), str(amount), reason])
    with open(os.path.join('log', dispence_logfile_path), 'a+') as dispence_log_file:
        dispence_log_file.write(string + "\n")
        dispence_log_file.flush()


def daily_log(basetime):
    """ 餌やりの日計ログ 記入事項：「日付, 量(粒)/報酬として,量(粒)/精算として」 """
    feeds = pd.DataFrame(columns=["date", "feed_num", "reason"])
    if os.path.exists(os.path.join("log", dispence_logfile_path)):
        feeds = pd.read_csv(os.path.join("log", dispence_logfile_path), names=["date", "feed_num", "reason"],
                            parse_dates=[0])
    # this "basetime" means end point
    feed_dataframe = feeds[(datetime.now() > feeds.date) & (feeds.date > basetime - timedelta(days=1))]
    feed_list = [str(datetime.now()),
                 "reward",
                 str(feed_dataframe.groupby("reason").sum().loc["reward", "feed_num"] if any(
                     feed_dataframe.reason.isin(["reward"])) else 0)
                 ]
    payoff_dataframe = feeds[(datetime.now() < feeds.date) & (feeds.date > basetime + timedelta(days=1))]
    feed_list += ["payoff",
                  str(payoff_dataframe.groupby("reason").sum().loc["payoff", "feed_num"] if any(
                      payoff_dataframe.reason.isin(["payoff"])) else 0)]
    string = ','.join(feed_list)
    with open(os.path.join('log', daily_logfile_path), 'a+') as daily_dispence_log_file:
        daily_dispence_log_file.write(string + "\n")
        daily_dispence_log_file.flush()


def file_setup(mouse_no):
    global logfile_path, dispence_logfile_path, nosepoke_logfile_path, settings_logfile_path, daily_logfile_path
    logfile_path = logfile_path.format(mouse_no.zfill(3))
    dispence_logfile_path = dispence_logfile_path.format(mouse_no.zfill(3))
    daily_logfile_path = daily_logfile_path.format(mouse_no.zfill(3))
    nosepoke_logfile_path = nosepoke_logfile_path.format(mouse_no.zfill(3))
    settings_logfile_path = settings_logfile_path.format(mouse_no.zfill(3))
    shutil.copyfile(setting_file, os.path.join('log', settings_logfile_path))


def select_preview_payoff():
    # read csv
    with open(os.path.join("log", dispence_logfile_path), 'a+') as dispense_log_file:
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


def calc_todays_feed(basetime):
    # basetime: it's upper
    if not os.path.exists(dispence_logfile_path):
        return 0
    feeds_today_df = pd.read_csv(os.path.join("log", dispence_logfile_path), names=["date", "feed_num", "reason"],
                                 parse_dates=[0])
    feeds_today_df = feeds_today_df[
        ((basetime > feeds_today_df.date) & (feeds_today_df.date > basetime - timedelta(days=1)))]
    return feeds_today_df.groupby("reason").sum().loc["reward", "feed_num"]


if __name__ == "__main__":
    file_setup("3")
    magagine_log("test")
    all_nosepoke_log(3, "test")
    export("test", 1, 1, "reward", "3/5")
    export("test", 1, 1, "time over", "3/5")
    export("test", 1, 1, "reward", 7)
    export("test", 1, 1, "time over", None)
