#
from datetime import datetime, timedelta
from typing import Union

import pandas as pd
import numpy as np
from scipy.stats import entropy
import sys
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.markers as markers
import os
import csv
import time
from pyentrp import entropy as pent
import seaborn as sns
from pathlib import Path

def get_status(mouse_id) -> pd.DataFrame:
    file = "./data/sync/no{:03d}_action.csv".format(mouse_id)
    data = pd.read_csv(file, names=["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"],
                       parse_dates=[0])
    if isinstance(data.iloc[0].timestamps, str):
        data = pd.read_csv(file, parse_dates=[0])
        data.columns = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
    data = data[["timestamps", "event_type", "task", "hole_no"]]
    tasks = data.task.unique().tolist()

    event_times = pd.pivot_table(data[data.event_type.isin(["reward", "failure", "time over"])], index="event_type",
                                 columns="task", aggfunc="count").timestamps
    task_duration = pd.DataFrame(data.groupby("task").timestamps.max() - data.groupby("task").timestamps.min())
    task_duration.timestamps = task_duration.timestamps / np.timedelta64(1, 'h')
    ret_val = event_times.append(task_duration.T).fillna(0)
    ret_val = ret_val.rename(index={"timestamps": "duration in hours"})
    # 列の順番をtaskをやった順でソート
    ret_val = ret_val.loc[:, tasks]
    # ret_val.to_csv("./data/no{:03d}_summary.csv".format(no))
    pd.options.display.precision = 1
    # print(ret_val)

    # entropy
    entropy_series = pd.Series(name='entropy')
    for last_task in tasks:
        tasks_data = data[data.task.isin([last_task])]
        task_data_tmp = tasks_data
        tasks_data = pd.pivot_table(tasks_data[tasks_data.event_type.isin(["reward","failure","time over"])],index="event_type",columns="hole_no",aggfunc="count").timestamps.fillna(0)
        tasks_data.loc["total_trials"] = tasks_data.sum()
        #print(tasks_data)
        choice_list = task_data_tmp[task_data_tmp.event_type.isin(["reward", "failure"])]["hole_no"].head(300).to_list()
        ent = pent.shannon_entropy(choice_list)
        entropy_series[last_task] = ent
        #print(f"[{last_task}] entropy = {ent:.3f}, length={len(choice_list): d}")
    ret_val = ret_val.append(entropy_series)

    ret_val = ret_val.T
#    print(ret_val.index)

    return ret_val

def write_report_csv(mouse_id):
    print(f"[{mouse_id}]")
    status_df = get_status(mouse_id)
#    print(status_df)
    ldf = status_df.tail(1)

    task = ldf.index[0]
    duration = ldf['duration in hours'].iloc[-1]
    try:
        reward = ldf['reward'].iloc[-1]
    except:
        reward = 0
    try:
        failure = ldf['failure'].iloc[-1]
    except:
        failure = 0
    try:
        timeover = ldf['time over'].iloc[-1]
    except:
        timeover = 0
    try:
        entropy = ldf['entropy'].iloc[-1]
    except:
        entropy = 0
    str = f"{task} {duration:.0f}h R{reward:.0f} F{failure:.0f} T{timeover:.0f} E{entropy:.2f}"
    print(str)
    f = open(f'./data/sync/report/{mouse_id:03d}.txt', 'w')
    f.write(str)
    f.close()

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1:
        for i in args[1:len(args)]:
            write_report_csv(int(i))
    else: # 最新5件
        p = Path('./data/sync/')
        files = list(p.glob('no???_action.csv'))
        for file in files[-6:-1]: # 最後の5データ
            mouse_id = int(file.stem[2:5])
            write_report_csv(mouse_id)

#        file_updates = {file_path: os.stat(file_path).st_mtime for file_path in files}
#        newst_file_path = max(file_updates, key=file_updates.get)
#        print(newst_file_path)
#        write_report_csv(int(i))
