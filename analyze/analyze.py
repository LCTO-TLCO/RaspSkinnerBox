import matplotlib as mpl
import pandas as pd
from pandas.core.window import Rolling
import numpy as np
import pyper
import os
import datetime
from scipy.stats import entropy


def read_data():
    data_file = "../RaspSkinnerBox/log/no004_action.csv"
    script_file = "./reshape_data.R"
    time_limit_delta = 60
    header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
    data = pd.read_csv(data_file, names=header, parse_dates=[0])
    data = data[data["event_type"].isin(["reward", "failure", "time over"])]
    data = data.reset_index()
    # task interval
    task_start_index = [0]
    for i in range(1, len(data)):
        if not data["task"][i] == data["task"][i - 1]:
            task_start_index.append(i)

    data["hole_correct"] = -1
    data["hole_incorrect"] = -1
    data["is_correct"] = -1
    data["is_incorrect"] = -1
    data["is_omission"] = -1
    data["cumsum_correct"] = -1
    data["cumsum_incorrect"] = -1
    data["cumsum_omission"] = -1
    data["cumsum_correct_taskreset"] = -1
    data["cumsum_incorrect_taskreset"] = -1
    data["cumsum_omission_taskreset"] = -1
    for hole_no in range(1, 9 + 1, 2):
        data["is_hole{}".format(str(hole_no))] = -1

    print(data)

    for i in range(0, len(data)):
        event_type = data["event_type"][i]
        print(event_type)
        # hole_information
        data["hole_correct"][i] = data["hole_no"][i] if event_type == "reward" else 0
        data["hole_incorrect"][i] = data["hole_no"][i] if event_type == "failure" else 0
        data["is_correct"][i] = 1 if event_type == "reward" else 0
        data["is_incorrect"][i] = 1 if event_type == "failure" else 0
        data["is_omission"][i] = 1 if event_type == "time over" else 0
        data["cumsum_correct"][i] = sum(data["is_correct"][0:i + 1])
        data["cumsum_incorrect"][i] = sum(data["is_incorrect"][0:i + 1])
        data["cumsum_omission"][i] = sum(data["is_omission"][0:i + 1])

        # is_holex
        for hole_no in range(1, 9 + 1, 2):
            data["is_hole{}".format(hole_no)][i] = 1 if str(hole_no) in data["hole_no"][i] else None
    # cumsum
    for i in range(0, len(task_start_index)):
        index_start = task_start_index[i]
        index_end = len(data)
        if i < len(task_start_index) - 1:
            index_end = task_start_index[i + 1]
        pre_correct = data["cumsum_correct"][index_start] if not i == 0 else 0
        pre_incorrect = data["cumsum_incorrect"][index_start] if not i == 0 else 0
        pre_omission = data["cumsum_omission"][index_start] if not i == 0 else 0
        data["cumsum_correct_taskreset"][index_start:index_end] = data["cumsum_correct"][index_start:index_end] - \
                                                                  pre_correct
        data["cumsum_incorrect_taskreset"][index_start:index_end] = data["cumsum_incorrect"][index_start:index_end] - \
                                                                    pre_incorrect
        data["cumsum_omission_taskreset"][index_start:index_end] = data["cumsum_omission"][index_start:index_end] - \
                                                                   pre_omission

    # entropy
    # def ent(x):
    #     return entropy(x, base=2)
    #
    # data["hole_choice_entropy"]
    #     (ent, data["hole_no"], 150)
    # # burst
    # data["burst_group"] = 1
    # for i in range(1, len(data)):
    #     if data["timestamps"][i] - data["timestamps"][i - 1] <= datetime.timedelta(seconds=60):
    #         data["burst_group"][i] = data["burst_group"][i - 1]
    #         continue
    #     data["burst_group"][i] = data["burst_group"][i - 1] + 1
    return data


if __name__ == "__main__":
    read_data().to_csv('./test.csv')
