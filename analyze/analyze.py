import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import math, string, sys, fileinput
from scipy.stats import entropy
from graph import graph

debug = True


class task_data:
    def __init__(self, mice: list, tasks, logpath):
        global debug
        self.data_file = ""
        self.data = None
        self.data_ci = None
        self.delta = None
        self.mouse_no = mice
        self.tasks = tasks
        self.probability = {}
        self.mice_task = {}
        self.task_prob = {}
        self.mice_delta = {}
        self.logpath = logpath
        print('reading data...', end='')
        if debug:
            for mouse_id in self.mouse_no:
                self.mice_task[mouse_id], self.probability[mouse_id], self.task_prob[mouse_id], self.mice_delta[
                    mouse_id] = self.dev_read_data(mouse_id)
        else:
            for mouse_id in self.mouse_no:
                self.data_file = "{}no{:03d}_action.csv".format(self.logpath, mouse_id)
                self.mice_task[mouse_id], self.probability[mouse_id], self.task_prob[mouse_id], self.mice_delta[
                    mouse_id] = \
                    self.read_data()
                self.export_csv(mouse_id)
        print('done')

    def read_data(self):
        header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
        task_prob = {}

        def rehash_session_id():
            data = pd.read_csv(self.data_file, names=header, parse_dates=[0], dtype={'hole_no': 'str'})
            id_col = data.filter(items=["session_id", "task", "event_type"])
            line_no = 0
            print("max_id_col:{}".format(len(id_col)))
            session_id = 0
            end_col_no = -1
            # TODO reward, failureのあとのtime overが同じsession_idを持っている?(ことがある?)
            # TODO この部分の処理が重い (while文内のどこか)
            while line_no < len(id_col) - 1:
                if id_col["task"].iloc[line_no] == "T0":
                    if id_col["event_type"].iloc[line_no] == "start":
                        id_col.session_id.iloc[line_no] = session_id
                        line_no = line_no + 1
                    id_col.iloc[line_no]["session_id"] = session_id
                    line_no = line_no + 1
                    session_id = session_id + 1
                    continue
                next_sessionstart_row = id_col[line_no + 1:][id_col["event_type"].isin(["start"])]["session_id"].head(1)
                if len(next_sessionstart_row) == 0:
                    end_col_no = -1
                else:
                    end_col_no = \
                        next_sessionstart_row.index[0] - 1 + 1
                id_col[line_no:end_col_no]["session_id"] = session_id
                line_no = end_col_no if not end_col_no == 0 else len(id_col)
                if end_col_no == -1:
                    id_col[end_col_no] = session_id
                session_id = session_id + 1
            data["session_id"] = id_col["session_id"]
            print("rehash done")
            return data

        def add_hot_vector():
            #    data = data[data["event_type"].isin(["reward", "failure", "time over"])]
            data = self.data
            data = data[data["event_type"].isin(["reward", "failure", "time over"])]
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
            data.loc[~data["event_type"].isin(["reward"]), 'hole_correct'] = np.nan
            data.loc[data["event_type"].isin(["failure"]), 'hole_incorrect'] = data["hole_no"]
            data.loc[~data["event_type"].isin(["failure"]), 'hole_incorrect'] = np.nan

            data.loc[data["event_type"].isin(["reward"]), 'is_correct'] = 1
            data.loc[~data["event_type"].isin(["reward"]), 'is_correct'] = np.nan
            data.loc[data["event_type"].isin(["failure"]), 'is_incorrect'] = 1
            data.loc[~data["event_type"].isin(["failure"]), 'is_incorrect'] = np.nan
            data.loc[data["event_type"].isin(["time over"]), 'is_omission'] = 1
            data.loc[~data["event_type"].isin(["time over"]), 'is_omission'] = np.nan

            data["cumsum_correct"] = data["is_correct"].cumsum(axis=0)
            data["cumsum_incorrect"] = data["is_incorrect"].cumsum(axis=0)
            data["cumsum_omission"] = data["is_omission"].cumsum(axis=0)

            for hole_no in range(1, 9 + 1, 2):
                data.loc[data['hole_no'].str.contains(str(hole_no)), "is_hole{}".format(hole_no)] = 1
                data.loc[~data['hole_no'].str.contains(str(hole_no)), "is_hole{}".format(hole_no)] = None

            self.data_ci = data[data["event_type"].isin(["reward", "failure"])]

            # cumsum
            # for i in range(0, len(task_start_index)):
            #     index_start = task_start_index[i]
            #     index_end = len(data)
            #     if i < len(task_start_index) - 1:
            #         index_end = task_start_index[i + 1]
            #     pre_correct = data["cumsum_correct"][index_start] if not i == 0 else 0
            #     pre_incorrect = data["cumsum_incorrect"][index_start] if not i == 0 else 0
            #     pre_omission = data["cumsum_omission"][index_start] if not i == 0 else 0
            #     # warning mettyaderu zone
            #     data["cumsum_correct_taskreset"][index_start:index_end] = data["cumsum_correct"][
            #                                                               index_start:index_end] - \
            #                                                               pre_correct
            #     data["cumsum_incorrect_taskreset"][index_start:index_end] = data["cumsum_incorrect"][
            #                                                                 index_start:index_end] - \
            #                                                                 pre_incorrect
            #     data["cumsum_omission_taskreset"][index_start:index_end] = data["cumsum_omission"][
            #                                                                index_start:index_end] - \
            #                                                                pre_omission
            def add_cumsum():
                data["cumsum_correct_taskreset"] = data["is_correct"].fillna(0)
                data["cumsum_incorrect_taskreset"] = data["is_incorrect"].fillna(0)
                data["cumsum_omission_taskreset"] = data["is_omission"].fillna(0)
                data["cumsum_correct_taskreset"] = data.groupby("task")["cumsum_correct_taskreset"].cumsum()
                data["cumsum_incorrect_taskreset"] = data.groupby("task")["cumsum_incorrect_taskreset"].cumsum()
                data["cumsum_omission_taskreset"] = data.groupby("task")["cumsum_omission_taskreset"].cumsum()

            add_cumsum()

            def min_max(x, axis=None):
                np.array(x)
                min = np.array(x).min(axis=axis)
                max = np.array(x).max(axis=axis)
                result = (x - min) / (max - min)
                return result

            # entropy
            # ent = [0] * 150
            # for i in range(0, len(data[data.event_type.str.contains('(reward|failure|time over)')]) - 150):
            #     # TODO entropyの計算にomissionは入れない
            #     denominator = 150.0  # sum([data["is_hole{}".format(str(hole_no))][i:i + 150].sum() for hole_no in range(1, 9 + 1, 2)])
            #     current_entropy = min_max(
            #         [data["is_hole{}".format(str(hole_no))][i:i + 150].sum() / denominator for hole_no in
            #          [1, 3, 5, 7, 9]])
            #     ent.append(entropy(current_entropy, base=2))
            # data["hole_choice_entropy"] = ent

            # burst
            # data["burst_group"] = 1
            # for i in range(1, len(data)):
            #     if data["timestamps"][i] - data["timestamps"][i - 1] <= datetime.timedelta(seconds=60):
            #         data["burst_group"][i] = data["burst_group"][i - 1]
            #         continue
            #     data["burst_group"][i] = data["burst_group"][i - 1] + 1
            return data

        def add_timedelta():
            data = self.data
            data = data[data.session_id.isin(
                data[(data["event_type"] == "reward") | (data["event_type"] == "failure")]["session_id"])]
            deltas = {}
            for task in self.tasks:
                delta_df = pd.DataFrame(
                    # columns=["type", "continuous_noreward_period", "reaction_time", "reward_latency"]
                )
                for session in data[data.task == task]["session_id"].unique():
                    # reaction time
                    current_target = data[(data["session_id"] == session) & (data["task"] == task)]
                    if bool(sum(current_target["event_type"].isin(["task called"]))):
                        task_call = current_target[current_target["event_type"] == "task called"]
                        task_end = current_target[current_target["event_type"] == "nose poke"]
                        reaction_time = task_end.timestamps.iloc[0] - task_call.timestamps.iloc[0]
                        # 連続無報酬期間
                        previous_reward = data[
                            (data["event_type"] == "reward") & (
                                    data["timestamps"] < task_call["timestamps"].iloc[0])].tail(1)
                        norewarded_time = task_call.timestamps.iloc[0] - previous_reward.timestamps.iloc[0]
                        correct_incorrect = "correct" if bool(
                            sum(current_target["event_type"].isin(["reward"]))) else "incorrect"
                        # df 追加
                        delta_df = delta_df.append(
                            {'type': 'reaction_time', 'continuous_noreward_period': norewarded_time,
                             'reaction_time': reaction_time, 'correct_incorrect': correct_incorrect},
                            ignore_index=True)
                    # reward latency
                    if bool(sum(current_target["event_type"].isin(["reward"]))) and bool(
                            sum(current_target["event_type"].isin(["task called"]))):
                        nose_poke = current_target[current_target["event_type"] == "nose poke"]
                        reward_latency = current_target[
                                             current_target["event_type"] == "magazine nose poked"].timestamps.iloc[0] - \
                                         nose_poke.timestamps.iloc[0]
                        previous_reward = data[
                            (data["event_type"] == "reward") & (
                                    data["timestamps"] < nose_poke["timestamps"].iloc[0])].tail(1)
                        norewarded_time = nose_poke.timestamps.iloc[0] - previous_reward.timestamps.iloc[0]
                        delta_df = delta_df.append(
                            {'type': 'reward_latency', 'continuous_noreward_period': norewarded_time,
                             'reward_latency': reward_latency}, ignore_index=True)
                deltas[task] = delta_df
            return deltas

        self.data = rehash_session_id()
        self.delta = add_timedelta()
        self.data = add_hot_vector()

        # action Probability
        after_c_all = float(len(self.data[self.data["is_correct"] == 1]))
        after_f_all = float(len(self.data[self.data["is_incorrect"] == 1]))
        after_c_starts = self.data[self.data["is_correct"] == 1]
        after_f_starts = self.data[self.data["is_incorrect"] == 1]
        after_c_all_task = {}
        after_f_all_task = {}
        after_c_starts_task = {}
        after_f_starts_task = {}
        for task in self.tasks:
            after_c_starts_task[task] = self.data[(self.data["is_correct"] == 1) & (self.data["task"] == task)]
            after_f_starts_task[task] = self.data[(self.data["is_incorrect"] == 1) & (self.data["task"] == task)]
            after_c_all_task[task] = float(len(after_c_starts_task[task]))
            after_f_all_task[task] = float(len(after_f_starts_task[task]))

        # after_o_all = len(data[data["event_type"] == "time over"])
        forward_trace = 5
        prob_index = ["c_same", "c_diff", "c_omit", "c_checksum", "f_same", "f_diff", "f_omit", "f_checksum",
                      "c_NotMax",
                      "f_NotMax", "o_NotMax"]
        probability = pd.DataFrame(columns=prob_index, index=range(1, forward_trace + 1)).fillna(0.0)

        # count
        def count_all():
            # correctスタート
            for idx, dt in after_c_starts.iterrows():
                is_continued = True
                for j in range(1, min(forward_trace, len(self.data) - idx)):
                    # 報酬を得たときと同じ選択(CF両方)をしたときの処理
                    if dt["hole_no"] == self.data_ci.shift(-j)["hole_no"][
                        idx] and is_continued:  # TODO omissionを除いてカウントしたい
                        probability["c_same"][j] = probability["c_same"][j] + 1
                    # omissionの場合
                    elif self.data.shift(-j)["is_omission"][idx]:
                        probability["c_omit"][j] = probability["c_omit"][j] + 1
                        # is_continued = False
                    elif dt["hole_no"] != self.data_ci.shift(-j)["hole_no"][idx] and is_continued:
                        probability["c_diff"][j] = probability["c_diff"][j] + 1
                    # 違うhole
            #            else:
            #                is_continued = False
            # incorrectスタート
            for idx, dt in after_f_starts.iterrows():
                is_continued = True
                for j in range(1, min(forward_trace, len(self.data) - idx)):
                    # 連続で失敗しているときの処理
                    if dt["hole_no"] == self.data_ci.shift(-j)["hole_no"][
                        idx] and is_continued:  # TODO omissionを除いてカウントしたい
                        probability["f_same"][j] = probability["f_same"][j] + 1
                    elif self.data.shift(-j)["is_omission"][idx] and is_continued:
                        probability["f_omit"][j] = probability["f_omit"][j] + 1
                    elif dt["hole_no"] != self.data.shift(-j)["hole_no"][idx] and not \
                            self.data.shift(-j)["is_omission"][
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
                    for j in range(1, min(forward_trace, len(self.data) - idx)):
                        # 報酬を得たときと同じ選択(CF両方)をしたときの処理
                        if dt["hole_no"] == self.data_ci.shift(-j)["hole_no"][
                            idx] and is_continued:  # TODO omissionを除いてカウントしたい
                            prob["c_same"][j] = prob["c_same"][j] + 1
                        # omissionの場合
                        elif self.data.shift(-j)["is_omission"][idx]:
                            prob["c_omit"][j] = prob["c_omit"][j] + 1
                            # is_continued = False
                        elif dt["hole_no"] != self.data.shift(-j)["hole_no"][idx] and is_continued:
                            prob["c_diff"][j] = prob["c_diff"][j] + 1
                        # 違うhole
                #            else:
                #                is_continued = False
                # incorrectスタート
                for idx, dt in after_f_starts_task[task].iterrows():
                    is_continued = True
                    for j in range(1, min(forward_trace, len(self.data) - idx)):
                        # 連続で失敗しているときの処理
                        if dt["hole_no"] == self.data_ci.shift(-j)["hole_no"][
                            idx] and is_continued:  # TODO omissionを除いてカウントしたい
                            prob["f_same"][j] = prob["f_same"][j] + 1
                        elif self.data.shift(-j)["is_omission"][idx] and is_continued:
                            prob["f_omit"][j] = prob["f_omit"][j] + 1
                        elif dt["hole_no"] != self.data.shift(-j)["hole_no"][idx] and not \
                                self.data.shift(-j)["is_omission"][
                                    idx] and is_continued:
                            prob["f_diff"][j] = prob["f_diff"][j] + 1
                            # is_continued = False
                        else:
                            is_continued = False
                # calculate
                prob["c_same"] = prob["c_same"] / after_c_all_task[task] if not after_c_all_task[
                                                                                    task] == 0 else 0.0  # TODO omissionを除いてカウントしたい ので母数を修正
                prob["c_diff"] = prob["c_diff"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_omit"] = prob["c_omit"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_checksum"] = prob["c_same"] + prob["c_diff"] + prob["c_omit"]
                prob["f_same"] = prob["f_same"] / after_f_all_task[task] if not after_f_all_task[
                                                                                    task] == 0 else 0.0  # TODO omissionを除いてカウントしたい ので母数を修正
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
        return self.data, probability, task_prob, self.delta

    def dev_read_data(self, mouse_no):
        task_prob = {}
        data = pd.read_csv('{}data/no{:03d}_{}_data.csv'.format(self.logpath, mouse_no, "all"))
        probability = pd.read_csv('{}data/no{:03d}_{}_prob.csv'.format(self.logpath, mouse_no, "all"))
        delta = {}
        for task in self.tasks:
            delta[task] = pd.read_csv('{}data/no{:03d}_{}_time.csv'.format(self.logpath, mouse_no, task))
            task_prob[task] = pd.read_csv('{}data/no{:03d}_{}_prob.csv'.format(self.logpath, mouse_no, task))
        return data, probability, task_prob, delta

    def export_csv(self, mouse_no):
        self.mice_task[mouse_no].to_csv('{}data/no{:03d}_{}_data.csv'.format(self.logpath, mouse_no, "all"))
        self.probability[mouse_no].to_csv('{}data/no{:03d}_{}_prob.csv'.format(self.logpath, mouse_no, "all"))
        for task in self.tasks:
            self.mice_delta[mouse_no][task].to_csv('{}data/no{:03d}_{}_time.csv'.format(self.logpath, mouse_no, task))
            self.task_prob[mouse_no][task].to_csv('{}data/no{:03d}_{}_prob.csv'.format(self.logpath, mouse_no, task))


# TODO Burst raster plot
# TODO 散布図,csv出力 連続無報酬期間 vs reaction time (タスクコールからnose pokeまでの時間 正誤両方)
# TODO 散布図,csv出力 連続無報酬期間 vs reward latency  (正解nose pokeからmagazine nose pokeまでの時間 正解のみ)

# TODO 1111*(正正正正) fig1={P(基点とsame), N数}, fig2={P(一つ前とsame), N数}, fig3={P(omission)}, fig4={P()}
# TODO 1110
# TODO 1101
# TODO 1100*
# TODO 1011
# TODO 1010*
# TODO 1001
# TODO 1000*
# TODO 0111*
# TODO 0110
# TODO 0101*
# TODO 0100
# TODO 0011
# TODO 0010
# TODO 0001*
# TODO 0000*(誤誤誤誤), 4bit固定ではなくn bit対応で構築(念のため過去の履歴がどこまで効くのか見たいので10bitとかでグラフは保存), *はグラフ表示
# TODO 個体毎と全個体 (n数が不足すると思われるため全個体分も必要)

# TODO 散布図,csv出力 連続無報酬期間 vs 区間Entropy (検討中)
# TODO 探索行動の短期指標を定義(Exploration Index 1, EI1) : 検討中


if __name__ == "__main__":
    # mice = [6, 7, 8, 11, 12, 13, 17]
    #    mice = [17]
    mice = [12]
    tasks = ["All5_30", "Only5_50", "Not5_Other30"]
    #    logpath = '../RaspSkinnerBox/log/'
    logpath = './'
    task = task_data(mice, tasks, logpath)
    graph_ins = graph(task, mice, tasks, logpath)
    # graph_ins.entropy_scatter()
    # graph_ins.nose_poke_raster()
    # graph_ins.same_plot()
    # graph_ins.omission_plot()
    # graph_ins.ent_raster_cumsum()
    # graph_ins.reward_latency_scatter()
    graph_ins.reaction_scatter()

    # TODO 複数マウスで同一figureにplotしてしまっているので、別figureをそれぞれ立ち上げて描画・保存

    print('hoge')
