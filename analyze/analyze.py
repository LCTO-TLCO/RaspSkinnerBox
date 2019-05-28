import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyper
import datetime

from scipy.stats import entropy


# from pandas.core.window import Rolling

class task_data:
    def __init__(self, mice: list, tasks):
        self.data_file = ""
        self.mouse_no = mice
        self.tasks = tasks
        self.probability = {}
        self.mice_task = {}
        self.task_prob = {}
        for mouse_id in self.mouse_no:
            self.data_file = "../RaspSkinnerBox/log/no{:03d}_action.csv".format(mouse_id)
            self.mice_task[mouse_id], self.probability[mouse_id], self.task_prob[mouse_id] = \
                self.read_data()
            self.export_csv(mouse_id)

    def read_data(self):
        header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
        # TODO session_idはCTRL-Dで止めたらresetされる、app側を変えるか?, その場合過去のlogを一括修正する必要あり
        task_prob = {}

        def add_hot_vector():
            #    data = data[data["event_type"].isin(["reward", "failure", "time over"])]
            data = pd.read_csv(self.data_file, names=header, parse_dates=[0], dtype={'hole_no': 'str'})
            data = data[data["event_type"].isin(["reward", "failure"])]
            data = data[data["task"].isin(self.tasks)]

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
                data["cumsum_correct_taskreset"][index_start:index_end] = data["cumsum_correct"][
                                                                          index_start:index_end] - \
                                                                          pre_correct
                data["cumsum_incorrect_taskreset"][index_start:index_end] = data["cumsum_incorrect"][
                                                                            index_start:index_end] - \
                                                                            pre_incorrect
                data["cumsum_omission_taskreset"][index_start:index_end] = data["cumsum_omission"][
                                                                           index_start:index_end] - \
                                                                           pre_omission

            # entropy
            ent = [0] * 150
            for i in range(0, len(data[data.event_type.str.contains('(reward|failure)')]) - 150):
                current_entropy = [data["is_hole{}".format(str(hole_no))][i:i+150].sum()/150.0 for hole_no in [1, 3, 5, 7, 9]]
                ent.append(entropy(current_entropy, base=2))
            data["hole_choice_entropy"] = ent

            # burst
            # data["burst_group"] = 1
            # for i in range(1, len(data)):
            #     if data["timestamps"][i] - data["timestamps"][i - 1] <= datetime.timedelta(seconds=60):
            #         data["burst_group"][i] = data["burst_group"][i - 1]
            #         continue
            #     data["burst_group"][i] = data["burst_group"][i - 1] + 1
            return data

        data = add_hot_vector()

        def add_timedelta():
            data = pd.read_csv(self.data_file, names=header, parse_dates=[0], dtype={'hole_no': 'str'})
            delta_df = pd.DataFrame(
                columns=["type", "continuous_noreward_period", "reaction_time", "reward_latency", "entropy"])
            for task in self.tasks:
                for session in data[data.task == task]["session_id"].unique():
                    # reaction time
                    current_target = data[(data["session_id"] == session) & (data["task"] == task)]
                    if "task called" in current_target["event_type"]:
                        task_call = current_target[current_target["event_target"] == "task called"]
                        task_end = current_target[current_target["event_target"] == "nose poke"]
                        reaction_time = task_end.timestamps - task_call.timestamps
                        # 連続無報酬期間
                        previous_reward = data[
                            (data["event_type"] == "reward") & (data["timestamps"] < task_call["timestamps"])].tail(1)
                        norewarded_time = task_call.timestamps - previous_reward.timestamps
                        correct_incorrect = "correct" if current_target["event_type"].isin(["reward"]) else "incorrect"
                        # df 追加
                        delta_df.append(pd.DataFrame(
                            {'type': 'reaction_time', 'continuous_noreward_period': norewarded_time,
                             'reaction_time': reaction_time, 'correct_incorrect': correct_incorrect}))
                    # reward latency
                    if "reward" in current_target["event_type"] and "task called" in current_target["event_type"]:
                        nose_poke = current_target[current_target["event_type"] == "nose poke"]
                        reward_latency = current_target[
                                             current_target["event_type"] == "magazine nose poked"].timestamps - \
                                         nose_poke.timestamps
                        previous_reward = data[
                            (data["event_type"] == "reward") & (data["timestamps"] < nose_poke["timestamps"])].tail(1)
                        norewarded_time = nose_poke.timestamps - previous_reward.timestamps
                        delta_df.append(
                            pd.DataFrame({'type': 'reward_latency', 'continuous_noreward_period': norewarded_time,
                                          'reward_latency': reward_latency}))
                    # entropy

        # add_timedelta()

        # action Probability
        after_c_all = float(len(data[data["is_correct"] == 1]))
        after_f_all = float(len(data[data["is_incorrect"] == 1]))
        after_c_starts = data[data["is_correct"] == 1]
        after_f_starts = data[data["is_incorrect"] == 1]
        after_c_all_task = {}
        after_f_all_task = {}
        after_c_starts_task = {}
        after_f_starts_task = {}
        for task in self.tasks:
            after_c_starts_task[task] = data[(data["is_correct"] == 1) & (data["task"] == task)]
            after_f_starts_task[task] = data[(data["is_incorrect"] == 1) & (data["task"] == task)]
            after_c_all_task[task] = float(len(after_c_starts_task))
            after_f_all_task[task] = float(len(after_f_starts_task))

        # after_o_all = len(data[data["event_type"] == "time over"])
        forward_trace = 10
        prob_index = ["c_same", "c_diff", "c_omit", "c_checksum", "f_same", "f_diff", "f_omit", "f_checksum",
                      "c_NotMax",
                      "f_NotMax", "o_NotMax"]
        probability = pd.DataFrame(columns=prob_index, index=range(1, forward_trace)).fillna(0.0)

        # count
        # correctスタート
        def count_all():
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
                    elif dt["hole_no"] != data.shift(-j)["hole_no"][idx] and not data.shift(-j)["is_omission"][
                        idx] and is_continued:
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

        def count_task() -> dict:
            for task in self.tasks:
                prob = pd.DataFrame(columns=prob_index, index=range(1, forward_trace)).fillna(0.0)
                for idx, dt in after_c_starts_task[task].iterrows():
                    is_continued = True
                    for j in range(1, min(forward_trace, len(data) - idx)):
                        # 報酬を得たときと同じ選択(CF両方)をしたときの処理
                        if dt["hole_no"] == data.shift(-j)["hole_no"][idx] and is_continued:
                            prob["c_same"][j] = prob["c_same"][j] + 1
                        # omissionの場合
                        elif data.shift(-j)["is_omission"][idx]:
                            prob["c_omit"][j] = prob["c_omit"][j] + 1
                            # is_continued = False
                        elif dt["hole_no"] != data.shift(-j)["hole_no"][idx] and is_continued:
                            prob["c_diff"][j] = prob["c_diff"][j] + 1
                        # 違うhole
                #            else:
                #                is_continued = False
                # incorrectスタート
                for idx, dt in after_f_starts_task[task].iterrows():
                    is_continued = True
                    for j in range(1, min(forward_trace, len(data) - idx)):
                        # 連続で失敗しているときの処理
                        if dt["hole_no"] == data.shift(-j)["hole_no"][idx] and is_continued:
                            prob["f_same"][j] = prob["f_same"][j] + 1
                        elif data.shift(-j)["is_omission"][idx] and is_continued:
                            prob["f_omit"][j] = prob["f_omit"][j] + 1
                        elif dt["hole_no"] != data.shift(-j)["hole_no"][idx] and not data.shift(-j)["is_omission"][
                            idx] and is_continued:
                            prob["f_diff"][j] = prob["f_diff"][j] + 1
                            # is_continued = False
                        else:
                            is_continued = False
                # calculate
                prob["c_same"] = prob["c_same"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_diff"] = prob["c_diff"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_omit"] = prob["c_omit"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_checksum"] = prob["c_same"] + prob["c_diff"] + prob["c_omit"]
                prob["f_same"] = prob["f_same"] / after_f_all_task[task] if not after_f_all_task[task] == 0 else 0.0
                prob["f_diff"] = prob["f_diff"] / after_f_all_task[task] if not after_f_all_task[task] == 0 else 0.0
                prob["f_omit"] = prob["f_omit"] / after_f_all_task[task] if not after_f_all_task[task] == 0 else 0.0
                prob["f_checksum"] = prob["f_same"] + prob["f_diff"] + prob["f_omit"]

                # prob$c_NotMax %/=% after_c_all
                # prob$f_NotMax %/=% after_f_all
                # prob$o_NotMax %/=% after_o_all
                # append
                task_prob[task] = prob

        count_all()
        count_task()
        return data, probability, task_prob

    def export_csv(self, mouse_no):
        for task in self.tasks:
            self.mice_task[mouse_no].to_csv('../RaspSkinnerBox/log/no{:03d}_{}_data.csv'.format(mouse_no, task))
            self.probability[mouse_no].to_csv('../RaspSkinnerBox/log/no{:03d}_{}_prob.csv'.format(mouse_no, task))


# TODO Burst raster plot
# TODO R plot(Entropy, Raster, Correct/Incorrect/Omission) 移植
# TODO 散布図,csv出力 連続無報酬期間 vs reaction time (タスクコールからnose pokeまでの時間 正誤両方)
# TODO 散布図,csv出力 連続無報酬期間 vs reward latency  (正解nose pokeからmagazine nose pokeまでの時間 正解のみ)
# TODO 散布図,csv出力 連続無報酬期間 vs 区間Entropy
# TODO 探索行動の短期指標を定義(Exploration Index 1, EI1) : 検討中

class graph:
    def __init__(self, task_datas, mice, tasks):
        plt.style.use("ggplot")
        font = {'family': 'meiryo'}
        mpl.rc('font', **font)
        self.data = task_datas
        self.mice = mice
        self.tasks = tasks

    # data plot
    def burst_nosepoke(self):
        for mouse_id in self.mice:
            burst_nosepoke = plt.figure()
            burst_ax = burst_nosepoke.add_subplot(1, 1, 1)
            labels = ["correct", "incorrect", "omission"]
            # flags = data.loc[:, data.colums.str.match("is_[(omission|correct|incorrect)")]
            datasets = [(self.data.mice_task[mouse_id][self.data.mice_task[mouse_id]
                                                       ["is_{}".format(flag)] == 1]) for flag in labels]
            for dt, la in zip(datasets, labels):
                burst_ax.scatter(dt['session_id'], dt['is_hole1'] * 1, color="blue")
                burst_ax.scatter(dt['session_id'], dt['is_hole3'] * 2, color="blue")
                burst_ax.scatter(dt['session_id'], dt['is_hole5'] * 3, color="blue")
                burst_ax.scatter(dt['session_id'], dt['is_hole7'] * 4, color="blue")
                burst_ax.scatter(dt['session_id'], dt['is_hole9'] * 5, color="blue")
                burst_ax.scatter(dt['session_id'], dt['is_omission'] * 0, color="red")
            burst_ax.set_xlabel("time/sessions")
            burst_ax.set_ylabel("hole dots")
            plt.ylim(-1, 6)
            plt.show()

    def same_plot(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        for mouse_id in self.mice:
            for task in self.tasks:
                # P(same) plot
                plt.subplot(1, len(self.tasks), self.tasks.index(task) + 1)
                plt.plot(self.data.task_prob[mouse_id][task]["c_same"], label="correct")
                plt.plot(self.data.task_prob[mouse_id][task]["f_same"], label="incorrect")
                plt.ioff()
                plt.ylim(0, 1)
                if self.tasks.index(task) == 0:
                    plt.ylabel('P (same choice)')
                    plt.legend()
                plt.xlabel('Trial')
                plt.title('{:03} {}'.format(mouse_id, task))
                plt.show()

            plt.savefig('../RaspSkinnerBox/log/no{:03d}_prob.png'.format(mouse_id))
        print('done')

    def CFO_plot(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        for mouse_id in self.mice:
            self.data.task_prob[mouse_id].plot.scatter(x='session_id', y='is_correct')

    def burst_raster(self):
        None

    def reaction_scatter(self):
        None

    def reward_latency_scatter(self):
        None

    def entropy_scatter(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        for mouse_id in self.mice:
            self.data.mice_task[mouse_id].plot.line(x='session_id', y='hole_choice_entropy')


if __name__ == "__main__":
    # mice = [6, 7, 8, 11, 12, 13]
    mice = [14]
    tasks = ["All5_30", "Only5_50", "Not5_Other30"]
    # tasks = ["test"]
    task = task_data(mice, tasks)
    graph_ins = graph(task, mice, tasks)
    graph_ins.burst_nosepoke()
