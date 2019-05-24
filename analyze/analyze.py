import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyper
import datetime


# from scipy.stats import entropy
# from pandas.core.window import Rolling


def read_data():
    data_file = "../RaspSkinnerBox/log/test_action.csv"
    script_file = "./reshape_data.R"
    time_limit_delta = 60
    header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
    data = pd.read_csv(data_file, names=header, parse_dates=[0], dtype={'hole_no': 'str'})
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

    # print(data)

    # hole_information
    # warning mettyaderu zone
    # SettingWithCopyWarning
    data.loc[data["event_type"].isin(["reward"]), 'hole_correct'] = data["hole_no"]
    data.loc[~data["event_type"].isin(["reward"]), 'hole_correct'] = 0
    data.loc[data["event_type"].isin(["failure"]), 'hole_incorrect'] = data["hole_no"]
    data.loc[~data["event_type"].isin(["failure"]), 'hole_incorrect'] = 0

    data.loc[data["event_type"].isin(["reward"]), 'is_correct'] = 1
    data.loc[~data["event_type"].isin(["reward"]), 'is_correct'] = 0
    data.loc[data["event_type"].isin(["failure"]), 'is_incorrect'] = 1
    data.loc[~data["event_type"].isin(["failure"]), 'is_incorrect'] = 0
    data.loc[data["event_type"].isin(["time over"]), 'is_omission'] = 1
    data.loc[~data["event_type"].isin(["time over"]), 'is_omission'] = 0

    data["cumsum_correct"] = data["is_correct"].cumsum(axis=0)
    data["cumsum_incorrect"] = data["is_incorrect"].cumsum(axis=0)
    data["cumsum_omission"] = data["is_omission"].cumsum(axis=0)

    for hole_no in range(1, 9 + 1, 2):
        data.loc[data['hole_no'].str.contains(str(hole_no)), "is_hole{}".format(hole_no)] = 1
        data.loc[~data['hole_no'].str.contains(str(hole_no)), "is_hole{}".format(hole_no)] = None

    # cumsum
    for i in range(0, len(task_start_index)):
        index_start = task_start_index[i]
        index_end = len(data)
        if i < len(task_start_index) - 1:
            index_end = task_start_index[i + 1]
        pre_correct = data["cumsum_correct"][index_start] if not i == 0 else 0
        pre_incorrect = data["cumsum_incorrect"][index_start] if not i == 0 else 0
        pre_omission = data["cumsum_omission"][index_start] if not i == 0 else 0
        # warning mettyaderu zone
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

    # action Probability
    after_c_all = float(len(data[data["is_correct"] & data.shift()["is_omission"]]))
    after_f_all = float(len(data[data["is_incorrect"] == 1]))
    after_c_starts = data[data["is_correct"] & data.shift()["is_omission"]]
    after_f_starts = data[(data["is_incorrect"] == 1) & (data.shift()["is_omission"]==1)]

    # after_o_all = len(data[data["event_type"] == "time over"])
    prob_index = ["c_same", "f_diff", "c_omit", "f_omit", "c_NotMax", "f_NotMax", "o_NotMax"]
    probability = pd.DataFrame(columns=prob_index, index=range(1, 5)).fillna(0.0)
    # count
    for idx, dt in after_c_starts.iterrows():
        is_continued = True
        for j in range(1, min(5, len(data) - idx)):
            # 連続で報酬を得ているときの処理
            if dt["hole_no"] == data.shift(-j)["hole_no"][idx] and is_continued:
                probability["c_same"][j] = probability["c_same"][j] + 1
            elif data.shift(-j)["is_omission"][idx]:
                probability["c_omit"][j] = probability["c_omit"][j] + 1
                is_continued = False
            else:
                is_continued = False
    for idx, dt in after_f_starts.iterrows():
        is_continued = True
        for j in range(1, min(5, len(data) - idx)):
            if data.shift(-j)["is_omission"][idx] and is_continued:
                probability["f_omit"][j] = probability["f_omit"][j] + 1
            elif not dt["hole_no"] == data.shift(-j)["hole_no"][idx]:
                probability["f_diff"][j] = probability["f_diff"][j] + 1
                is_continued = False
            else:
                is_continued = False
    # calculate
    probability["c_same"] = probability["c_same"] / after_c_all if not after_c_all == 0 else 0.0
    probability["f_diff"] = probability["f_diff"] / after_f_all if not after_f_all == 0 else 0.0
    probability["c_omit"] = probability["c_omit"] / after_c_all if not after_c_all == 0 else 0.0
    probability["f_omit"] = probability["f_omit"] / after_f_all if not after_f_all == 0 else 0.0

    # prob$c_NotMax %/=% after_c_all
    # prob$f_NotMax %/=% after_f_all
    # prob$o_NotMax %/=% after_o_all

    return [data, probability]


def graph(data):
    # define
    plt.style.use("ggplot")
    font = {'family': 'meiryo'}
    mpl.rc('font', **font)

    # data plot
    def burst_nosepoke():
        burst_nosepoke = plt.figure()
        burst_ax = burst_nosepoke.add_subplot(1, 1, 1)
        labels = ["correct", "incorrect", "omission"]
        # flags = data.loc[:, data.colums.str.match("is_[(omission|correct|incorrect)")]
        datasets = [(data[data["is_{}".format(flag)] == 1]) for flag in labels]
        for dt, la in zip(datasets, labels):
            burst_ax.scatter(dt['timestamps'], dt['is_hole1'] * 1)
            burst_ax.scatter(dt['timestamps'], dt['is_hole3'] * 2)
            burst_ax.scatter(dt['timestamps'], dt['is_hole5'] * 3)
            burst_ax.scatter(dt['timestamps'], dt['is_hole7'] * 4)
            burst_ax.scatter(dt['timestamps'], dt['is_hole9'] * 5)
            burst_ax.scatter(dt['timestamps'], dt['is_omission'] * 0)

        burst_ax.set_xlabel("time/sessions")
        burst_ax.set_ylabel("hole dots")
        plt.show()

    def aa():
        None

    burst_nosepoke()


if __name__ == "__main__":
    data, prob = read_data()
    data.to_csv('./test.csv')
    prob.to_csv('./prob.csv')
    # graph(data)
