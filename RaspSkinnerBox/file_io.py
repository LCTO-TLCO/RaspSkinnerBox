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
critlogfile_path = 'no{}_critics.csv'
ex_flow = OrderedDict({})
# ex_flow = OrderedDict()
ex_flow.update(json.load(open(setting_file, "r"), object_pairs_hook=OrderedDict))


class Pycolor:
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


colors = {'time over': Pycolor.PURPLE,
          'reward': Pycolor.GREEN,
          'failure': Pycolor.YELLOW,
          'task called': Pycolor.WHITE,
          }
integers = {'1': Pycolor.CYAN,
            '3': Pycolor.GREEN,
            '5': Pycolor.RED,
            '7': Pycolor.YELLOW,
            '9': Pycolor.BLUE}


def set_dir():
    pass
    # if (os.path.basename(os.getcwd()) == "log"):
    #     return
    # os.chdir("./log")


def export(task_no: str, session_no: int, times: int, event_type: str, hole_no=0):
    # わかんないときは一つ前の行をコピーしてくる
    if task_no == "":
        task_no = last_str("task") if last_str("task") else task_no
    if session_no == -1:
        session_no = last_str("session") if isinstance(last_str("session"), int) else session_no
    if times == -1:
        times = last_str("rewardnum") if isinstance(last_str("rewardnum"), int) else times
    logstring = ','.join([str(datetime.now()), task_no, str(session_no), str(times), event_type])
    with open(os.path.join("log", logfile_path), 'a+') as logfile:
        head = ["Timestamps", "task", "session", "rewardnum", "action", "cond"]
        if not os.path.getsize(os.path.join("log", logfile_path)):
            logfile.write(",".join(head) + "\n")
        logfile.write(",".join([logstring, str(hole_no)]) + "\n")
        logfile.flush()
    print(add_color(logstring, str(hole_no)))


def export_crit(task: str, trials: int, accuracy: float, omission: float, correct: int):
    logstring = ','.join(
        [str(datetime.now()), task, str(trials), str(accuracy), str(omission), str(correct)])
    with open(os.path.join("log", critlogfile_path), 'a+') as logfile:
        head = ["Timestamps", "task", "trials", "accuracy", "omission", "correct"]
        if not os.path.getsize(os.path.join("log", critlogfile_path)):
            logfile.write(",".join(head) + "\n")
        logfile.write(logstring + "\n")
        logfile.flush()


def add_color(string: str, integer: str):
    global colors, integers
    for keyword, new_color in colors.items():
        string = string.replace(keyword, "".join([new_color, keyword, Pycolor.END]))
    retval = ""
    for st in integer:
        if st in integers.keys():
            retval += st.replace(st, "".join([integers[st], st, Pycolor.END]))
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
    feed_dataframe = feeds[(basetime > feeds.date) & (feeds.date > basetime - timedelta(days=1))]
    feed_list = [str(datetime.now()),
                 "reward",
                 str(feed_dataframe.groupby("reason").sum().loc["reward", "feed_num"] if any(
                     feed_dataframe.reason.isin(["reward"])) else 0)
                 ]
    payoff_dataframe = feeds[(basetime > feeds.date) & (feeds.date > basetime - timedelta(days=1))]
    feed_list += ["payoff",
                  str(payoff_dataframe.groupby("reason").sum().loc["payoff", "feed_num"] if any(
                      payoff_dataframe.reason.isin(["payoff"])) else 0)]
    string = ','.join(feed_list)
    with open(os.path.join('log', daily_logfile_path), 'a+') as daily_dispence_log_file:
        daily_dispence_log_file.write(string + "\n")
        daily_dispence_log_file.flush()


def file_setup(mouse_no):
    global logfile_path, dispence_logfile_path, nosepoke_logfile_path, settings_logfile_path, daily_logfile_path, critlogfile_path
    logfile_path = logfile_path.format(mouse_no.zfill(3))
    dispence_logfile_path = dispence_logfile_path.format(mouse_no.zfill(3))
    daily_logfile_path = daily_logfile_path.format(mouse_no.zfill(3))
    nosepoke_logfile_path = nosepoke_logfile_path.format(mouse_no.zfill(3))
    settings_logfile_path = settings_logfile_path.format(mouse_no.zfill(3))
    shutil.copyfile(setting_file, os.path.join('log', settings_logfile_path))
    critlogfile_path = critlogfile_path.format(mouse_no.zfill(3))
    read_rehash_dataframe()


# def select_preview_payoff():
#     # read csv
#     with open(os.path.join("log", dispence_logfile_path), 'a+') as dispense_log_file:
#         data = csv.reader(dispense_log_file)
#         # select cols
#         preview_payoff_time = -1
#         if datetime.now().hour < 10:
#             preview_payoff_time = datetime(datetime.today().year, datetime.today().month, datetime.today().day - 1,
#                                            10, 0, 0)
#         else:
#             preview_payoff_time = datetime(datetime.today().year, datetime.today().month, datetime.today().day,
#                                            10, 0, 0)

def last_str(name):
    if not os.path.exists(os.path.join("log", logfile_path)):
        return 0
    else:
        logfile = pd.read_csv(os.path.join("log", logfile_path), parse_dates=[0])
        last = logfile.tail(1)
        if last.empty:
            return 0
        return last[name].to_list()[0]


def last_session_id(task=""):
    if not os.path.exists(os.path.join("log", logfile_path)):
        return 0
    else:
        logfile = pd.read_csv(os.path.join("log", logfile_path), parse_dates=[0])
        logfile = logfile[logfile.task.isin([task])]
        last = logfile.tail(1)
        if last.empty:
            return 0
        return int(last.session.to_list()[0]) + 1


def read_rehash_dataframe():
    if not os.path.exists(os.path.join("log", logfile_path)):
        return

    feeds = pd.read_csv(os.path.join("log", logfile_path), parse_dates=[0])

    def rehash_session_id():
        data = feeds
        print("max_id_col:{}".format(len(data)))

        def remove_terminate(index):
            if data.at[index, "action"] == data.at[index + 1, "action"] and data.at[index, "action"] == "start":
                data.drop(index, inplace=True)

        def rehash(x_index):
            id = data.at[data.index[max(x_index - 1, 0)], "session"]
            if data.at[data.index[x_index], "task"] == "T0" or x_index == 0:
                if (x_index == 0 or data.shift(1).at[data.index[x_index], "action"] == "start") and \
                        len(data[:x_index][data.session == 0]) == 0:
                    data.at[x_index, "session"] = 0
                    return 0
                data.at[x_index, "session"] = id + 1
                return id + 1
            if data.at[data.index[x_index], "action"] == "start":
                data.at[x_index, "session"] = id + 1
                return id + 1
            else:
                data.at[x_index, "session"] = id
                return id

        list(map(remove_terminate, data.index[:-1]))
        data.reset_index(drop=True, inplace=True)
        data["session"] = list(map(rehash, data.index))
        data = data[data.session.isin(data.session[data.action.isin(["reward", "failure", "premature", "time over"])])]
        data.reset_index(drop=True, inplace=True)
        data["session"] = list(map(rehash, data.index))
        return data

    feeds = rehash_session_id()
    feeds.to_csv(os.path.join("log", logfile_path), index=False)
    # return feeds


def select_last_session_log(session_duration=20, task=""):
    if not os.path.exists(os.path.join("log", logfile_path)):
        return {"accuracy": 0, "omission": 0, "correct": 0, "session": 0}
    else:
        feeds = pd.read_csv(os.path.join("log", logfile_path), parse_dates=[0])
        feeds = feeds[
            (feeds.action.isin(["reward", "failure", "premature", "time over"]) &
             (feeds.task.isin([task])))]
        if feeds.empty:
            return {"accuracy": 0, "omission": 0, "correct": 0, "session": 0}
        ret_val = {
            "accuracy":
                (feeds.action.isin(["reward"]).rolling(session_duration, min_periods=1).sum().iloc[-1] /
                 feeds.action.rolling(session_duration, min_periods=1).count().iloc[-1]),
            "omission": (feeds.action.isin(["time over"]).rolling(session_duration, min_periods=1).sum().iloc[-1] /
                         feeds.action.rolling(session_duration, min_periods=1).count().iloc[-1]),
            "correct": feeds.action.isin(["reward"]).count(),
            "session": feeds.action.count()
        }
        return ret_val


def error_log(error):
    logging.exception(error)
    raise


def callback_rising(channel):
    all_nosepoke_log(channel, "RISING")


def callback_falling(channel):
    all_nosepoke_log(channel, "FALLING")


def all_nosepoke_log(channel: int, event_type: str):
    # export(task_no: str, session_no: int, times: int, event_type: str, hole_no=0)
    export("", -1, -1, event_type, channel)


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
