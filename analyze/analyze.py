import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyper
import datetime


# from scipy.stats import entropy
# from pandas.core.window import Rolling


def read_data(data_file, mouse_no, task):
    header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"] # TODO session_idはCTRL-Dで止めたらresetされる、app側を変えるか?, その場合過去のlogを一括修正する必要あり
    data = pd.read_csv(data_file, names=header, parse_dates=[0], dtype={'hole_no': 'str'})

#    data = data[data["event_type"].isin(["reward", "failure", "time over"])]
    data = data[data["event_type"].isin(["reward", "failure"])]
    data = data[data["task"].isin([task])]

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
    after_c_all = float(len(data[data["is_correct"] == 1]))
    after_f_all = float(len(data[data["is_incorrect"] == 1]))
    after_c_starts = data[data["is_correct"] == 1]
    after_f_starts = data[data["is_incorrect"] == 1]

    # after_o_all = len(data[data["event_type"] == "time over"])
    forward_trace = 10
    prob_index = ["c_same", "c_diff", "c_omit", "c_checksum", "f_same", "f_diff", "f_omit", "f_checksum", "c_NotMax", "f_NotMax", "o_NotMax"]
    probability = pd.DataFrame(columns=prob_index, index=range(1, forward_trace)).fillna(0.0)
    # count
    # correctスタート
    for idx, dt in after_c_starts.iterrows():
        is_continued = True
        for j in range(1, min(forward_trace, len(data) - idx)):
            # 報酬を得たときと同じ選択(CF両方)をしたときの処理
            if dt["hole_no"] == data.shift(-j)["hole_no"][idx] and is_continued:
                probability["c_same"][j] = probability["c_same"][j] + 1
            # omissionの場合
            elif data.shift(-j)["is_omission"][idx]:
                probability["c_omit"][j] = probability["c_omit"][j] + 1
                # is_continued = False
            elif dt["hole_no"] != data.shift(-j)["hole_no"][idx] and is_continued:
                probability["c_diff"][j] = probability["c_diff"][j] + 1
            # 違うhole
#            else:
#                is_continued = False
    # incorrectスタート
    for idx, dt in after_f_starts.iterrows():
        is_continued = True
        for j in range(1, min(forward_trace, len(data) - idx)):
            # 連続で失敗しているときの処理
            if dt["hole_no"] == data.shift(-j)["hole_no"][idx] and is_continued:
                probability["f_same"][j] = probability["f_same"][j] + 1
            elif data.shift(-j)["is_omission"][idx] and is_continued:
                probability["f_omit"][j] = probability["f_omit"][j] + 1
            elif dt["hole_no"] != data.shift(-j)["hole_no"][idx] and not data.shift(-j)["is_omission"][idx] and is_continued:
                probability["f_diff"][j] = probability["f_diff"][j] + 1
                # is_continued = False
            else:
                is_continued = False
    # calculate
    probability["c_same"] = probability["c_same"] / after_c_all if not after_c_all == 0 else 0.0
    probability["c_diff"] = probability["c_diff"] / after_c_all if not after_c_all == 0 else 0.0
    probability["c_omit"] = probability["c_omit"] / after_c_all if not after_c_all == 0 else 0.0
    probability["c_checksum"] = probability["c_same"] + probability["c_diff"] + probability["c_omit"]
    probability["f_same"] = probability["f_same"] / after_f_all if not after_f_all == 0 else 0.0
    probability["f_diff"] = probability["f_diff"] / after_f_all if not after_f_all == 0 else 0.0
    probability["f_omit"] = probability["f_omit"] / after_f_all if not after_f_all == 0 else 0.0
    probability["f_checksum"] = probability["f_same"] + probability["f_diff"] + probability["f_omit"]

    # prob$c_NotMax %/=% after_c_all
    # prob$f_NotMax %/=% after_f_all
    # prob$o_NotMax %/=% after_o_all

    return [data, probability]

# TODO R plot(Entropy, Raster, Correct/Incorrect/Omission) 移植
# TODO 散布図,csv出力 連続無報酬期間 vs reaction time (タスクコールからnose pokeまでの時間 正誤両方)
# TODO 散布図,csv出力 連続無報酬期間 vs reward latency  (正解nose pokeからmagazine nose pokeまでの時間 正解のみ)
# TODO 散布図,csv出力 連続無報酬期間 vs 区間Entropy
# TODO 探索行動の短期指標を定義(Exploration Index 1, EI1) : 検討中

def graph(data, prob, mouse_id, task):
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
    # burst_nosepoke()



if __name__ == "__main__":
    mice = [6, 7, 8, 11, 12, 13]
    tasks = ["All5_30", "Only5_50", "Not5_Other30"]

    for mouse_id in mice:

        fig = plt.figure(figsize=(15, 8), dpi=100) # TODO view分離

        i = 0
        for task in tasks:
            i = i + 1
            print('mouse id={:03} task={}'.format(mouse_id, task))
            data_file = "../RaspSkinnerBox/log/no{:03d}_action.csv".format(mouse_id)
            print('analyzing ...', end=' ')
            data, prob = read_data(data_file, mouse_id, task)  # TODO mouse_id, task毎にdataを管理したい(data,probのリスト化?)
            print('done')
            print('csv writing ...', end=' ')
            data.to_csv('../RaspSkinnerBox/log/no{:03d}_{}_data.csv'.format(mouse_id, task))
            prob.to_csv('../RaspSkinnerBox/log/no{:03d}_{}_prob.csv'.format(mouse_id, task))
            print('done')

            print('plotting ...', end=' ')

            # P(same) plot
            plt.subplot(1, 3, i)
            plt.plot(prob["c_same"], label="correct")
            plt.plot(prob["f_same"], label="incorrect")
            plt.ioff()
            plt.ylim(0, 1)
            if i == 1:
                plt.ylabel('P (same choice)')
                plt.legend()
            plt.xlabel('Trial')
            plt.title('{:03} {}'.format(mouse_id, task))
            plt.show()

            # TODO plotを関数化 (data, probを算出する部分(model)とplot(view)を分離する)
            print('done')

        plt.savefig('../RaspSkinnerBox/log/no{:03d}_prob.png'.format(mouse_id)) # TODO view分離

