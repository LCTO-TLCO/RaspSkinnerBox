from datetime import datetime, timedelta
from typing import Union

import pandas as pd
import numpy as np
import math
from scipy.stats import entropy
# from graph import rasp_graph
import sys
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.markers as markers
import os

# debug = True
debug = False


# TODO graph classの内容をscratch内で実行？　

class task_data:
    def __init__(self, mice: list, tasks, logpath):
        global debug
        self.data_file = ""
        self.data = None
        self.data_ci = None
        self.delta = None
        self.mouse_no = mice
        self.tasks = tasks
        self.pattern_prob = {}
        self.probability = None
        self.mice_task = None
        self.task_prob = {}
        self.mice_delta = {}
        self.entropy_analyze = None
        self.mice_entropy = None
        self.logpath = logpath
        self.session_id = 0
        self.burst_id = 0
        self.data_not_omission = None
        self.fig_prob_tmp = None
        self.fig_prob = {}
        self.bit = 4

        print('reading data...', end='')

        def append_dataframe(to: Union[pd.DataFrame, dict, None], add: Union[pd.DataFrame, dict, None], mouse_id: int,
                             task=None, fig_num=None):
            if isinstance(add, dict):

                # 二回目の場合Fig確定
                if not isinstance(task, type(None)):
                    # print(add)
                    ret_val = {}
                    [ret_val.update(append_dataframe(to, add[fig], mouse_id, task=task, fig_num=fig)) for fig in
                     ["fig1", "fig2", "fig3"]]
                    return {task: ret_val}
                # taskごと
                # to;dict, add:dict[task]
                for task, add_dict in add.items():
                    to.update(append_dataframe(to, add_dict, mouse_id, task=task))
                return to
            if isinstance(to, dict):
                # Fig二回目入力
                if not isinstance(fig_num, type(None)):
                    return {fig_num: add.assign(mouse_id=mouse_id)}
                to[task] = add.assign(mouse_id=mouse_id)
                return to
            if isinstance(to, type(None)):
                return add.assign(mouse_id=mouse_id)
            else:
                return to.append(add.assign(mouse_id=mouse_id), ignore_index=True)

        if debug:
            for mouse_id in self.mouse_no:
                print('mouse_id={}'.format(mouse_id))
                self.mice_task[mouse_id], self.probability[mouse_id], self.task_prob[mouse_id], self.mice_delta[
                    mouse_id], self.fig_prob[mouse_id] = self.dev_read_data(mouse_id)
                # tmp = self.dev_read_data(mouse_id)
                # self.mice_task = append_dataframe(self.mice_task, tmp[0], mouse_id)
                # self.probability = append_dataframe(self.probability, tmp[1], mouse_id)
                # self.task_prob = append_dataframe(self.task_prob, tmp[2], mouse_id)
                # self.mice_delta = append_dataframe(self.mice_delta, tmp[3], mouse_id)
                # # append_dataframe(self.fig_prob, tmp[4], mouse_id)
                # self.fig_prob[mouse_id] = self.fig_prob[mouse_id].append(tmp[4])
                # self.pattern_prob = append_dataframe(self.pattern_prob, tmp[5], mouse_id)
                # TODO entropy_analyze
        else:
            for mouse_id in self.mouse_no:
                print('mouse_id={}'.format(mouse_id))
                # self.data_file = "{}no{:03d}_action.csv".format(self.logpath, mouse_id)
                # self.mice_task[mouse_id], self.probability[mouse_id], self.task_prob[mouse_id], self.mice_delta[
                #     mouse_id], self.fig_prob[mouse_id], self.pattern_prob[mouse_id] = self.read_data()
                self.data_file = os.path.join(self.logpath, "no{:03d}_action.csv".format(mouse_id))
                tmp = self.read_data()
                self.mice_task = append_dataframe(self.mice_task, tmp[0], mouse_id)
                self.probability = append_dataframe(self.probability, tmp[1], mouse_id)
                self.task_prob = append_dataframe(self.task_prob, tmp[2], mouse_id)
                self.mice_delta = append_dataframe(self.mice_delta, tmp[3], mouse_id)
                # append_dataframe(self.fig_prob, tmp[4], mouse_id)
                self.fig_prob = append_dataframe(self.fig_prob, tmp[4], mouse_id)
                self.pattern_prob = append_dataframe(self.pattern_prob, tmp[5], mouse_id)
                self.mice_entropy = append_dataframe(self.mice_entropy, tmp[6], mouse_id)
                self.export_csv(mouse_id)
        print('done')

    def read_data(self):

        def rehash_session_id():
            data = pd.read_csv(self.data_file, names=header, parse_dates=[0], dtype={'hole_no': 'str'})
            self.session_id = 0
            print("max_id_col:{}".format(len(data)))

            def remove_terminate(index):
                if data.at[index, "event_type"] == data.at[index + 1, "event_type"] and data.at[
                    index, "event_type"] == "start":
                    data.drop(index, inplace=True)

            def rehash(x_index):
                if data.at[data.index[x_index], "task"] == "T0":
                    if (x_index == 0 or data.shift(1).at[data.index[x_index], "event_type"] == "start") and \
                            len(data[:x_index][data.session_id == 0]) == 0:
                        self.session_id = 0
                        return 0
                    self.session_id = self.session_id + 1
                    return self.session_id
                if data.at[data.index[x_index], "event_type"] == "start":
                    self.session_id = self.session_id + 1
                    return self.session_id
                else:
                    return self.session_id

            list(map(remove_terminate, data.index[:-1]))
            data.reset_index(drop=True, inplace=True)
            data["session_id"] = list(map(rehash, data.index))
            data = data[data.session_id.isin(data.session_id[data.event_type.isin(["reward", "failure", "time over"])])]
            data.reset_index(drop=True, inplace=True)
            self.session_id = 0
            data["session_id"] = list(map(rehash, data.index))
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))
            return data

        def add_timedelta():
            data = self.data
            data = data[data.session_id.isin(data[data.event_type.isin(['reward', 'failure'])]["session_id"])]
            deltas = {}
            for task in self.tasks:
                def calculate(session):
                    delta_df = pd.DataFrame()
                    # reaction time
                    current_target = data[data.session_id.isin([session])]
                    if bool(sum(current_target["event_type"].isin(["task called"]))):
                        task_call = current_target[current_target["event_type"] == "task called"]
                        task_end = current_target[current_target["event_type"] == "nose poke"]
                        reaction_time = task_end.at[task_end.index[0], "timestamps"] - task_call.at[
                            task_call.index[0], "timestamps"]
                        # 連続無報酬期間
                        previous_reward = data[
                            (data["event_type"] == "reward") & (
                                    data["timestamps"] < task_call.at[task_call.index[0], "timestamps"])].tail(1)
                        norewarded_time = task_call.at[task_call.index[0], "timestamps"] - previous_reward.at[
                            previous_reward.index[0], "timestamps"]
                        correct_incorrect = "correct" if bool(
                            sum(current_target["event_type"].isin(["reward"]))) else "incorrect"
                        # df 追加
                        delta_df = delta_df.append(
                            {'type': 'reaction_time',
                             'noreward_duration_sec': pd.to_timedelta(norewarded_time) / np.timedelta64(1, 's'),
                             'reaction_time_sec': pd.to_timedelta(reaction_time) / np.timedelta64(1, 's'),
                             'correct_incorrect': correct_incorrect},
                            ignore_index=True)
                    # reward latency
                    if bool(sum(current_target["event_type"].isin(["reward"]))) and bool(
                            sum(current_target["event_type"].isin(["task called"]))):
                        nose_poke = current_target[current_target["event_type"] == "nose poke"]
                        reward_latency = current_target[current_target["event_type"] == "magazine nose poked"]
                        reward_latency = reward_latency.at[reward_latency.index[0], "timestamps"] - \
                                         nose_poke.at[nose_poke.index[0], "timestamps"]
                        previous_reward = data[
                            (data["event_type"] == "reward") & (
                                    data["timestamps"] < nose_poke.at[nose_poke.index[0], "timestamps"])].tail(1)
                        norewarded_time = nose_poke.at[nose_poke.index[0], "timestamps"] - previous_reward.at[
                            previous_reward.index[0], "timestamps"]
                        delta_df = delta_df.append(
                            {'type': 'reward_latency',
                             'noreward_duration_sec': pd.to_timedelta(norewarded_time) / np.timedelta64(1, 's'),
                             'reward_latency_sec': pd.to_timedelta(reward_latency) / np.timedelta64(1, 's')
                             }, ignore_index=True)
                    return delta_df

                delta_df = data[data.task == task].session_id.drop_duplicates().map(calculate)
                deltas[task] = pd.concat(list(delta_df), sort=False)
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))
            return deltas

        def add_hot_vector():
            #    data = data[data["event_type"].isin(["reward", "failure", "time over"])]
            data = self.data
            # data = data[data[".seevent_type"].isin(["reward", "failure", "time over"])]
            # data = data[data["task"].isin(self.tasks)]

            data = data.reset_index(drop=True)
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
                data.loc[data['hole_no'] == str(hole_no), "is_hole{}".format(hole_no)] = 1
                data.loc[~(data['hole_no'] == str(hole_no)), "is_hole{}".format(hole_no)] = None

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

            # burst
            # data["burst_group"] = 1
            # for i in range(1, len(data)):
            #     if data["timestamps"][i] - data["timestamps"][i - 1] <= datetime.timedelta(seconds=60):
            #         data["burst_group"][i] = data["burst_group"][i - 1]
            #         continue
            #     data["burst_group"][i] = data["burst_group"][i - 1] + 1
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))
            return data

        def calc_entropy(section=150):
            data = self.data[self.data.event_type.isin(['reward', 'failure'])]

            def min_max(x, axis=None):
                np.array(x)
                min = np.array(x).min(axis=axis)
                max = np.array(x).max(axis=axis)
                result = (x - min) / (max - min)
                return result

            # entropy
            ent = [np.nan] * section
            for i in range(0, len(data) - section):
                denominator = float(section)
                # sum([data["is_hole{}".format(str(hole_no))][i:i + 150].sum() for hole_no in range(1, 9 + 1, 2)])
                current_entropy = min_max(
                    [data["is_hole{}".format(str(hole_no))][i:i + section].sum() /
                     denominator for hole_no in [1, 3, 5, 7, 9]])
                ent.append(entropy(current_entropy, base=2))
            # region Description
            # data[data.event_type.isin(['reward', 'failure'])]["hole_choice_entropy"] = ent
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))
            return pd.DataFrame(ent).fillna(0.0).values.tolist()
            # endregion

        def count_task() -> dict:
            dc = self.data[self.data["event_type"].isin(["reward", "failure"])]
            # dc = self.data[self.data["event_type"].isin(["reward", "failure", "time over"])]

            dc = dc.reset_index()

            after_c_all_task = {}
            after_f_all_task = {}

            after_c_starts_task = {}
            after_f_starts_task = {}

            prob_index = ["c_same", "c_diff", "c_omit", "c_checksum", "f_same", "f_diff", "f_omit", "f_checksum",
                          "c_NotMax",
                          "f_NotMax", "o_NotMax"]
            forward_trace = 10

            for task in self.tasks:
                after_c_starts_task[task] = dc[(dc["is_correct"] == 1) & (dc["task"] == task)]
                after_f_starts_task[task] = dc[(dc["is_incorrect"] == 1) & (dc["task"] == task)]
                after_c_all_task[task] = float(len(after_c_starts_task[task]))
                after_f_all_task[task] = float(len(after_f_starts_task[task]))

                prob = pd.DataFrame(columns=prob_index, index=range(1, forward_trace)).fillna(0.0)
                # correctスタート
                for idx, dt in after_c_starts_task[task].iterrows():
                    for j in range(1, min(forward_trace, len(dc) - idx)):
                        #                    for j in range(1, min(forward_trace, len(self.data_cio) - idx)):
                        # 報酬を得たときと同じ選択(CF両方)をしたときの処理
                        if dt["hole_no"] == dc["hole_no"][idx + j]:
                            prob["c_same"][j] = prob["c_same"][j] + 1
                        # omissionの場合
                        elif dc["is_omission"][idx + j] == 1:
                            prob["c_omit"][j] = prob["c_omit"][j] + 1
                        elif dt["hole_no"] != dc["hole_no"][idx + j]:
                            prob["c_diff"][j] = prob["c_diff"][j] + 1

                # incorrectスタート
                for idx, dt in after_f_starts_task[task].iterrows():
                    for j in range(1, min(forward_trace, len(dc) - idx)):
                        #                    for j in range(1, min(forward_trace, len(self.data_cio) - idx)):
                        if dt["hole_no"] == dc["hole_no"][idx + j]:
                            prob["f_same"][j] = prob["f_same"][j] + 1
                        elif dc["is_omission"][idx + j] == 1:
                            prob["f_omit"][j] = prob["f_omit"][j] + 1
                        elif dt["hole_no"] != dc["hole_no"][idx + j]:
                            prob["f_diff"][j] = prob["f_diff"][j] + 1

                # calculate
                prob["c_same"] = prob["c_same"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_diff"] = prob["c_diff"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_omit"] = prob["c_omit"] / after_c_all_task[task] if not after_c_all_task[task] == 0 else 0.0
                prob["c_checksum"] = prob["c_same"] + prob["c_diff"] + prob["c_omit"]
                prob["f_same"] = prob["f_same"] / after_f_all_task[task] if not after_f_all_task[task] == 0 else 0.0
                prob["f_diff"] = prob["f_diff"] / after_f_all_task[task] if not after_f_all_task[task] == 0 else 0.0
                prob["f_omit"] = prob["f_omit"] / after_f_all_task[task] if not after_f_all_task[task] == 0 else 0.0
                prob["f_checksum"] = prob["f_same"] + prob["f_diff"] + prob["f_omit"]

                task_prob[task] = prob
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))

        # TODO 結構な確率でエラー吐く
        def analyze_pattern(bit=self.bit):
            fig_prob = {}
            pattern_range = range(0, pow(2, bit))
            for task in self.tasks:
                pattern[task] = {}
                fig_prob[task] = {"fig1": pd.DataFrame(columns=["{:b}".format(i).zfill(bit) for i in pattern_range]
                                                       ).fillna(0.0),
                                  "fig2": pd.DataFrame(columns=["{:b}".format(i).zfill(bit) for i in pattern_range]
                                                       ).fillna(0.0),
                                  "fig3": pd.DataFrame(columns=["{:b}".format(i).zfill(bit) for i in pattern_range],
                                                       ).fillna(0.0)}
                data = self.data[
                    (self.data.task == task) & (
                        self.data.event_type.isin(["reward", "failure", "time over"]))].reset_index(drop=True)
                data_ci = data[data.event_type.isin(["reward", "failure"])].reset_index(drop=True)
                # search pattern

                f_pattern_matching = lambda x: sum([
                    (not np.isnan(data_ci.at[x + (bit - i - 1), "is_correct"])) * pow(2, i)
                    for i in range(0, bit)])
                pattern[task] = data_ci[:-(bit - 1)].assign(pattern=data_ci[:-(bit - 1)].index.map(f_pattern_matching))
                # count

                f_same_base = lambda x: [data_ci.at[data_ci[data_ci.session_id == x].index[0], "hole_no"] == \
                                         data_ci.at[data_ci[data_ci.session_id == x].index[0] + idx, "hole_no"] for idx
                                         in range(1, bit)]
                f_same_prev = lambda x: [data_ci.at[data_ci[data_ci.session_id == x].index[0] + idx - 1, "hole_no"] == \
                                         data_ci.at[data_ci[data_ci.session_id == x].index[0] + idx, "hole_no"] for idx
                                         in range(1, bit)]
                f_omit = lambda x: [bool(data.at[data[data.session_id == x].index[0] + idx, "is_omission"]) for idx in
                                    range(1, bit)]
                functions = lambda x: [f_same_base(x), f_same_prev(x), f_omit(x)]
                # pattern count -> probability
                for pat_tmp in pattern_range:
                    f_p = pd.DataFrame(list(pattern[task][pattern[task].pattern == pat_tmp].session_id.map(functions)),
                                       columns=["fig1", "fig2", "fig3"]).fillna(0.0)
                    if len(f_p):
                        fig_prob[task]["fig1"]["{:b}".format(pat_tmp).zfill(bit)] = pd.DataFrame(
                            list(f_p.fig1)).sum().fillna(0.0) / len(pattern[task][pattern[task].pattern == pat_tmp])
                        fig_prob[task]["fig2"]["{:b}".format(pat_tmp).zfill(bit)] = pd.DataFrame(
                            list(f_p.fig2)).sum().fillna(0.0) / len(pattern[task][pattern[task].pattern == pat_tmp])
                        fig_prob[task]["fig3"]["{:b}".format(pat_tmp).zfill(bit)] = pd.DataFrame(
                            list(f_p.fig3)).sum().fillna(0.0) / len(pattern[task][pattern[task].pattern == pat_tmp])
                    else:
                        for figure in list(f_p.columns):
                            fig_prob[task][figure]["{:b}".format(pat_tmp).zfill(bit)] = fig_prob[task][figure][
                                "{:b}".format(pat_tmp).zfill(bit)].fillna(0.0)
                    for figure in list(f_p.columns):
                        # fig_prob[task][figure]["{:04b}".format(pat_tmp)].append(pd.Series([len(
                        #     pattern[task][pattern[task].pattern == pat_tmp])], index="n"))
                        fig_prob[task][figure].at["n", "{:b}".format(pat_tmp).zfill(bit)] = len(
                            pattern[task][pattern[task].pattern == pat_tmp])
                # self.data[self.data.task == task].loc["pattern"] = pattern[task].pattern
            # save
            self.fig_prob_tmp = fig_prob
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))
            return pattern

        def burst():
            def calc_burst(session):
                if session == 0:
                    self.burst_id = 0
                    return self.burst_id
                if data.at[data.index[data.session_id == session][0], "timestamps"] - \
                        data.at[data.index[data.session_id == session - 1][0], "timestamps"] >= timedelta(
                    seconds=60):
                    self.burst_id = self.burst_id + 1
                return self.burst_id

            data = self.data[self.data.event_type.isin(["reward", "failure", "time over"])]
            # self.data["burst"] = np.nan
            # self.data.loc[self.data.index[self.data.session_id == 0], "burst"] = 0
            self.data = self.data.merge(
                pd.DataFrame({"session_id": self.data.session_id.unique(),
                              "burst": list(map(calc_burst, self.data.session_id.unique()))}),
                on="session_id", how="left")
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))

        def entropy_analyzing(section=10, bit=self.bit):
            data = self.data[(self.data.event_type.isin(["reward", "failure"])) & (self.data.task.isin(self.tasks))]
            entropy_df = data[
                ["session_id", "task", "entropy_{}".format(section), "entropy_after_{}".format(section), "pattern"]]
            count_correct = lambda pat: np.nan if np.isnan(pat) else "{:b}".format(int(pat)).zfill(bit).count("1")
            entropy_df["correctnum_{}bit".format(bit)] = list(map(count_correct, entropy_df.pattern))
            # entropy_df["mouse_no"] = self.mouse_no
            print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))
            return entropy_df

        # main
        header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
        pattern = {}
        task_prob = {}
        self.data = rehash_session_id()
        self.data = add_hot_vector()
        self.data_ci = self.data
        self.data.loc[
            self.data.index[self.data.event_type.isin(['reward', 'failure'])], "hole_choice_entropy"] = calc_entropy()
        # ent_section = 10
        # self.data.loc[
        #     self.data.index[self.data.event_type.isin(['reward', 'failure'])], "entropy_10"] = calc_entropy(ent_section)
        # self.data.loc[
        #     self.data.index[self.data.event_type.isin(['reward', 'failure'])], "entropy_after_10"] = \
        #     self.data.loc[self.data.index[self.data.event_type.isin(
        #         ['reward', 'failure'])], "entropy_10"][(ent_section + self.bit - 1):].to_list() + \
        #     ([np.nan] * (ent_section + self.bit - 1))
        ent_section = 50
        self.data.loc[
            self.data.index[self.data.event_type.isin(['reward', 'failure'])], "entropy_{}".format(
                ent_section)] = calc_entropy(ent_section)
        self.data.loc[
            self.data.index[self.data.event_type.isin(['reward', 'failure'])], "entropy_after_{}".format(ent_section)] = \
            self.data.loc[self.data.index[self.data.event_type.isin(
                ['reward', 'failure'])], "entropy_{}".format(ent_section)][(ent_section + self.bit - 1):].tolist() + \
            ([np.nan] * (ent_section + self.bit - 1))
        self.delta = add_timedelta()
        self.data_not_omission = self.data[
            ~self.data.session_id.isin(self.data.session_id[self.data.event_type.isin(["time over"])])]

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

        #        count_all()
        count_task()
        # bit analyze
        pp = analyze_pattern(self.bit)
        pp = pd.concat([pp[task].loc[:, pp[task].columns.isin(["session_id", "pattern"])] for task in self.tasks])
        self.data = pd.merge(self.data, pp, how='left')
        # 2 bit analyze
        pp = analyze_pattern(2)
        pp = pd.concat([pp[task].loc[:, pp[task].columns.isin(["session_id", "pattern"])] for task in self.tasks])
        pp = pp.rename(columns={"pattern": "pattern_2bit"})
        self.data = pd.merge(self.data, pp, how='left')
        burst()
        # entropy analyzing
        self.entropy_analyze = entropy_analyzing(section=50)
        # self.entropy_analyze.concat(entropy_analyzing(section=50))
        return self.data, probability, task_prob, self.delta, self.fig_prob_tmp, pattern, self.entropy_analyze

    def dev_read_data(self, mouse_no):
        task_prob = {}
        delta = {}
        fig_prob = {}
        pattern_prob = {}
        data = pd.read_csv(os.path.join(self.logpath, 'data/no{:03d}_{}_data.csv'.format(mouse_no, "all")))
        probability = pd.read_csv(os.path.join(self.logpath, 'data/no{:03d}_{}_prob.csv'.format(mouse_no, "all")))

        for task in self.tasks:
            delta[task] = pd.read_csv(os.path.join(self.logpath, 'data/no{:03d}_{}_time.csv'.format(mouse_no, task)))
            task_prob[task] = pd.read_csv(
                os.path.join(self.logpath, 'data/no{:03d}_{}_prob.csv'.format(mouse_no, task)))
            fig_prob[task] = {}
            for fig_num in ["fig1", "fig2", "fig3"]:
                fig_prob[task][fig_num] = pd.read_csv(
                    os.path.join(self.logpath, 'data/no{:03d}_{}_{}_prob_fig.csv'.format(mouse_no, task, fig_num)),
                    index_col=0)
            pattern_prob[task] = pd.read_csv(
                os.path.join(self.logpath, 'data/no{:03d}_{}_pattern.csv'.format(mouse_no, task)))
        return data, probability, task_prob, delta, fig_prob, pattern_prob

    def export_csv(self, mouse_no):
        self.mice_task.to_csv(os.path.join(self.logpath, 'data/{}_data.csv'.format("all")))
        self.probability.to_csv(
            os.path.join(self.logpath, 'data/{}_prob.csv'.format("all")))
        for task in self.tasks:
            self.mice_delta[task].to_csv(os.path.join(self.logpath, 'data/{}_time.csv'.format(task)))
            # AttributeError: 'Series' object has no attribute 'type'
            reward_latency_data = self.mice_delta[task][self.mice_delta[task].type == "reward_latency"]
            reward_latency_data.to_csv(os.path.join(self.logpath, 'data/{}_rewardlatency.csv'.format(task)))
            self.task_prob[task].to_csv(os.path.join(self.logpath, 'data/{}_prob.csv'.format(task)))
            self.pattern_prob[task].to_csv(os.path.join(self.logpath, 'data/{}_pattern.csv'.format(task)))
            [self.fig_prob[task][fig_num].to_csv(
                os.path.join(self.logpath, 'data/prob_fig{}_{}.csv'.format(fig_num, task))) for
                fig_num in ["fig1", "fig2", "fig3"]]
            # pattern
            # [self.entropy_analyze[
            #      (self.entropy_analyze["correctnum_{}bit".format(10,self.bit)] == count) &
            #      (self.entropy_analyze["task"] == task)  # & (
            #      # self.entropy_analyze["mouse_no"] == mouse_no)
            #      ][10:-10].to_csv(
            #     '{}data/pattern_entropy/summary/no{:03d}_{}_entropy_pattern_count_{}_summary.csv'.format(
            #         self.logpath, mouse_no, task, int(count))) for count in
            #     self.entropy_analyze["correctnum_{}bit".format(10,self.bit)][
            #         ~np.isnan(self.entropy_analyze["correctnum_{}bit".format(10,self.bit)])].unique()]
            # [self.entropy_analyze[
            #      (self.entropy_analyze["pattern"] == pattern) &
            #      (self.entropy_analyze["task"] == task)  # & (
            #      # self.entropy_analyze["mouse_no"] == mouse_no)
            #      ][10:-10].to_csv(
            #     '{}data/pattern_entropy/no{:03d}_{}_entropy_pattern_{:04b}.csv'.format(
            #         self.logpath, mouse_no, task, int(pattern))) for
            #     pattern in self.data.pattern[~np.isnan(self.data.pattern)].unique()]
            [self.mice_entropy[(self.mice_entropy["task"] == task)  # & (
                 # self.entropy_analyze["mouse_no"] == mouse_no)
             ][50:-50][(self.mice_entropy["correctnum_{}bit".format(self.bit)] == count)].to_csv(
                '{}/data/pattern_entropy/summary/{}_entropy_pattern{:d}_count_{}_summary.csv'.format(
                    self.logpath, task, 50, int(count))) for count in
                # self.entropy_analyze["correctnum_{}bit".format(self.bit)][
                # ~np.isnan(self.entropy_analyze["correctnum_{}bit".format(self.bit)])].unique()]
                range(0, self.bit)]
            [self.mice_entropy[
                 (self.mice_entropy["task"] == task)  # & (
                 # self.entropy_analyze["mouse_no"] == mouse_no)
             ][50:-50][(self.mice_entropy["pattern"] == pattern)].to_csv(
                '{}/data/pattern_entropy/{}_entropy{:d}_pattern_{:04b}.csv'.format(
                    self.logpath, task, 50, int(pattern))) for
                pattern in self.data.pattern[~np.isnan(self.data.pattern)].unique()]

        print("{} ; {} done".format(datetime.now(), sys._getframe().f_code.co_name))


if __name__ == "__main__":
    print("{} ; started".format(datetime.now()))
    # mice = [2, 3, 6, 7, 8, 11, 12, 13, 14, 17, 18, 19]
    # error: 2,3,7,11,13,17,18
    # mice = [18]
    mice = [23]
    tasks = ["All5_30", "Only5_50", "Not5_Other30"]
    #    logpath = '../RaspSkinnerBox/log/'
    logpath = os.getcwd()
    # tdata = task_data(mice, tasks, logpath)
    # graph_ins = rasp_graph(tdata, mice, tasks, logpath)
    # graph_ins.entropy_scatter()
    # graph_ins.nose_poke_raster()
    # graph_ins.same_plot()
    # graph_ins.omission_plot()
    # graph_ins.ent_raster_cumsum()
    # graph_ins.reaction_scatter()
    # graph_ins.reaction_hist2d()
    # graph_ins.norew_reward_latency_scatter()
    # graph_ins.norew_reward_latency_hist2d()
    # graph_ins.prob_same_base()
    # graph_ins.prob_same_prev()
    # graph_ins.prob_omit()
    # graph_ins.next_10_ent()
    # graph_ins.norew_ent_10()
    # graph_ins.time_ent_10()
    # graph_ins.time_holeno_raster_burst()
    print("{} ; all done".format(datetime.now()))

    # TODO 同一burst内のデータでcountして平均表示


def view_averaged_prob_same_prev(tdata, mice, tasks):
    m = []
    t = []
    csame = []
    fsame = []

    for mouse_id in mice:
        for task in tasks:
            m += [mouse_id]
            t += [task]
            csame += [tdata.task_prob[task][tdata.task_prob[task].mouse_id == mouse_id]['c_same']]
            fsame += [tdata.task_prob[task][tdata.task_prob[task].mouse_id == mouse_id]['f_same']]

    after_prob_df = pd.DataFrame(
        data={'mouse_id': m, 'task': t, 'c_same': csame, 'f_same': fsame},
        columns=['mouse_id', 'task', 'c_same', 'f_same']
    )

    plt.style.use('default')
    fig = plt.figure(figsize=(8, 4), dpi=100)
    for task in tasks:
        plt.subplot(1, len(tasks), tasks.index(task) + 1)

        c_same = np.array(after_prob_df[after_prob_df['task'].isin([task])]['c_same'].to_list())
        c_same_avg = np.mean(c_same, axis=0)
        c_same_std = np.std(c_same, axis=0)
        c_same_var = np.var(c_same, axis=0)

        f_same = np.array(after_prob_df[after_prob_df['task'].isin([task])]['f_same'].to_list())
        f_same_avg = np.mean(f_same, axis=0)
        f_same_var = np.var(f_same, axis=0)

        xlen = len(c_same_avg)
        xax = np.array(range(1, xlen + 1))
        plt.plot(xax, c_same_avg, label="rewarded start")
        plt.errorbar(xax, c_same_avg, yerr=c_same_var, capsize=2, fmt='o', markersize=1, ecolor='black',
                     markeredgecolor="black", color='w', lolims=True)

        plt.plot(np.array(range(1, xlen + 1)), f_same_avg, label="no-rewarded start")
        plt.errorbar(xax, f_same_avg, yerr=f_same_var, capsize=2, fmt='o', markersize=1, ecolor='black',
                     markeredgecolor="black", color='w', uplims=True)

        # plt.ion()
        plt.xticks(np.arange(1, xlen + 1, 1))
        plt.xlim(0.5, xlen + 0.5)
        plt.ylim(0, 1.05)
        if tasks.index(task) == 0:
            plt.ylabel('P (same choice)')
            plt.legend()
        plt.xlabel('Trial')
        plt.title('{}'.format(task))
    # plt.savefig('fig/{}_prob_all4.png'.format(graph_ins.exportpath))
    plt.savefig('fig/prob_all4.png')
    plt.show()


# TODO task範囲の背景描画
# TODO 100ステップ移動平均を追加
def view_summary(tdata, mice, tasks):
    for mouse_id in mice:
        def plot(mdf, task="all"):

            # task color

            labels = ["incorrect", "correct", "omission"]
            df = mdf[mdf["event_type"].isin(["reward", "failure", "time over"])]

            # entropy
            fig = plt.figure(figsize=(15, 8), dpi=100)
            ax = [fig.add_subplot(3, 1, 1)]
            plt.plot(df.session_id, df['hole_choice_entropy'])
            plt.ylabel('Entropy (bit)')
            plt.xlim(df.session_id.min(), df.session_id.max())
            if task == "all":
                collection = collections.BrokenBarHCollection.span_where(df.session_id.to_numpy(), ymin=-100, ymax=100,
                                                                     where=(df.task.isin(tasks[0::2])),
                                                                     facecolor='pink', alpha=0.3)
                ax[0].add_collection(collection)

            # scatter
            ax.append(fig.add_subplot(3, 1, 2, sharex=ax[0]))
            colors = ["red", "blue", "black"]
            size = dict(zip(labels, [25, 50, 25]))
            pos = dict(zip(labels, ["bottom", "full", "bottom"]))
            datasets = ([mdf[mdf["is_{}".format(flag)] == 1] for flag in labels])
            for dt, la, cl in zip(datasets, labels, colors):
                marker = markers.MarkerStyle("|", pos[la])
                plt.scatter(dt.session_id, dt['is_hole1'] * 1, s=size[la], color=cl, marker=marker)
                plt.scatter(dt.session_id, dt['is_hole3'] * 2, s=size[la], color=cl, marker=marker)
                plt.scatter(dt.session_id, dt['is_hole5'] * 3, s=size[la], color=cl, marker=marker)
                plt.scatter(dt.session_id, dt['is_hole7'] * 4, s=size[la], color=cl, marker=marker)
                plt.scatter(dt.session_id, dt['is_hole9'] * 5, s=size[la], color=cl, marker=marker)
                plt.scatter(dt.session_id, dt['is_omission'] * 0, s=size[la], color=cl, marker=marker)
            plt.ylabel("Hole")
            if task == "all":
                collection = collections.BrokenBarHCollection.span_where(df.session_id.to_numpy(), ymin=-2, ymax=6,
                                                                     where=(df.task.isin(tasks[0::2])),
                                                                     facecolor='pink', alpha=0.3)
                ax[1].add_collection(collection)

            # cumsum
            ax.append(fig.add_subplot(3, 1, 3, sharex=ax[0]))
            plt.plot(df.session_id, df['cumsum_correct_taskreset'])
            plt.plot(df.session_id, df['cumsum_incorrect_taskreset'])
            plt.plot(df.session_id, df['cumsum_omission_taskreset'])
            # plt.xlim(0, df.session_id.max())
            plt.ylabel('Cumulative')
            plt.xlabel('Trial')
            if task == "all":
                collection = collections.BrokenBarHCollection.span_where(df.session_id.to_numpy(), ymin=-20, ymax=1000,
                                                                     where=(df.task.isin(tasks[0::2])),
                                                                     facecolor='pink', alpha=0.3)
                ax[2].add_collection(collection)

            plt.savefig('fig/no{:03d}_{}_summary.png'.format(mouse_id, task))
            plt.show()

        mdf = tdata.mice_task[tdata.mice_task.mouse_id == mouse_id]
        plot(mdf)
        # list(map(plot, [mdf[mdf.task == task] for task in tdata.tasks], tdata.tasks))


def view_trial_per_datetime(tdata, mice=[18], task="All5_30"):
    """ for debug """
    # for mouse_no in mice:
    data = tdata.data[
        (tdata.data.event_type.isin(["reward", "failure", "time over"]))
        & (tdata.data.task == task)
        # &(tdata.data.mouse_id == mouse_no)
        ].set_index("timestamps").resample("1H").sum()

    fig = plt.figure(figsize=(15, 8), dpi=100)
    data.plot.bar(y=["is_correct", "is_incorrect", "is_omission"], stacked=True)
    plt.show()


def view_scatter_vs_times_with_burst(tdata, mice=[18], task="All5_30", burst=1):
    """ fig1 B """
    for mouse_id in mice:

        labels = ["correct", "incorrect", "omission"]

        data = tdata.data.assign(
            timestamps=(tdata.data.timestamps - tdata.data.timestamps[0]).dt.total_seconds())  # [mouse_id]
        data = data[data["event_type"].isin(["reward", "failure", "time over"])]
        # data = data[data.burst.isin(data.burst.unique()[data.groupby("burst").burst.count() > burst])]
        burst_time = list(data.burst.unique()[data.groupby("burst").burst.count() > burst])
        fig = plt.figure(figsize=(15, 8), dpi=100)
        fig_subplot = fig.add_subplot(1, 1, 1)
        # plt.title('{:03} summary'.format(mouse_id))
        #    nose_poke_raster(mouse_id, fig.add_subplot(3, 1, 2))

        colors = ["blue", "red", "black"]
        for single_burst in burst_time:
            d = data[data.burst == single_burst]
            datasets = [(d[d["is_{}".format(flag)] == 1]) for flag in labels]
            for dt, la, cl in zip(datasets, labels, colors):
                plt.scatter(dt.timestamps, dt['is_hole1'] * 1, s=15, c=cl)
                plt.scatter(dt.timestamps, dt['is_hole3'] * 2, s=15, c=cl)
                plt.scatter(dt.timestamps, dt['is_hole5'] * 3, s=15, c=cl)
                plt.scatter(dt.timestamps, dt['is_hole7'] * 4, s=15, c=cl)
                plt.scatter(dt.timestamps, dt['is_hole9'] * 5, s=15, c=cl)
                plt.scatter(dt.timestamps, dt['is_omission'] * 0, s=15, c=cl)
            plt.ylabel("Hole")
            plt.xlim(d.timestamps.min() - 30, d.timestamps.max() + 30)
            plt.ylim(0, 5)
            #    plt.xlim(0, len(mdf))

            collection = collections.BrokenBarHCollection.span_where(data.timestamps.to_numpy(), ymin=0, ymax=5,
                                                                     where=(data.burst.isin(burst_time)),
                                                                     facecolor='pink', alpha=0.3)
            fig_subplot.add_collection(collection)
            # save
            # plt.show()
            burst_len = d.timestamps.count()
            if not os.path.isdir(os.path.join(os.getcwd(), "fig", "burst", "len" + str(burst_len))):
                os.mkdir(os.path.join(os.getcwd(), "fig", "burst", "len" + str(burst_len)))
            plt.savefig(os.path.join(os.getcwd(), 'fig', 'burst', "len" + str(burst_len),
                                     'no{:03d}_burst{}_hole_pasttime_burst.png'.format(mouse_id, single_burst)))

        # TODO task割表示
        # TODO rasterと他(cumsum, entropy)がずれている？


def view_trial_per_time(tdata, mice=[18], task="All5_30"):
    """ fig1 C """
    data = tdata.data[
        (tdata.data.event_type.isin(["reward", "failure", "time over"])) &
        (tdata.data.task == task)
        ].set_index("timestamps").resample("1H").sum()
    data = data.set_index(data.index.time).groupby(level=0).mean()
    fig = plt.figure(figsize=(15, 8), dpi=100)
    data.plot.bar(y=["is_correct", "is_incorrect", "is_omission"], stacked=True)
    plt.show()


def view_prob_same_choice_burst(tdata, mice, task, burst=1):
    """ fig4 """
    m = []
    t = []
    csame = []
    fsame = []

    tdata_cio = tdata.data[tdata.data.event_type.isin(["reward", "failure", "time over"])]
    data = tdata_cio[tdata_cio.burst.isin(tdata_cio.burst.unique()[tdata_cio.groupby("burst").burst.count() > burst])]
    for mouse_id in mice:
        for task in tasks:
            m += [mouse_id]
            t += [task]
            csame += [tdata.task_prob[mouse_id][task]['c_same']]
            fsame += [tdata.task_prob[mouse_id][task]['f_same']]

    after_prob_df = pd.DataFrame(
        data={'mouse_id': m, 'task': t, 'c_same': csame, 'f_same': fsame},
        columns=['mouse_id', 'task', 'c_same', 'f_same']
    )

    plt.style.use('default')
    fig = plt.figure(figsize=(8, 4), dpi=100)

    #
    plt.subplot(1, len(tasks), tasks.index(task) + 1)

    c_same = np.array(after_prob_df[after_prob_df['task'].isin([task])]['c_same'].to_list())

    c_same_avg = np.mean(c_same, axis=0)
    c_same_std = np.std(c_same, axis=0)
    c_same_var = np.var(c_same, axis=0)

    f_same = np.array(after_prob_df[after_prob_df['task'].isin([task])]['f_same'].to_list())
    f_same_avg = np.mean(f_same, axis=0)
    f_same_var = np.var(f_same, axis=0)


def view_only5_50(tdata, mice, task):
    """ fig5 A """
    pass


def view_not5_other30(tdata, mice, task):
    """ fig5 B """
    pass


def view_pattern_entropy_summary(tdata, mice, task=None):
    data = tdata.mice_entropy
    average_all = None
    for mouse_id in mice:
        data_tmp = data[mouse_id].groupby(
            ["task", "correctnum_{}bit".format(tdata.bit)])
        mean = data_tmp.mean().reset_index()
        sd = data_tmp.std().reset_index()
        data_tmp = pd.merge(mean, sd, on=["task", "correctnum_{}bit".format(tdata.bit)], suffixes=["_mean", "_sd"])
        data_tmp = data_tmp.loc[:, data_tmp.columns.str.startswith(("task", "correctnum", "entropy"))].assign(
            mouse_id=mouse_id)
        average_all = data_tmp if isinstance(average_all, type(None)) else average_all.append(data_tmp)
    for group_info, data_tmp in average_all.groupby(["task", "correctnum_{}bit".format(tdata.bit)]):
        fig, ax = plt.subplots(1, 1)
        # error bar
        # ax.errorbar(data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].columns.to_numpy().reshape(2),
        #             data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].to_numpy(),
        #             yerr=data_tmp.loc[:, data_tmp.columns.str.endswith("sd")].to_numpy(),
        #             ecolor="black")
        # ax.errorbar(data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].columns,
        #             data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].to_numpy().reshape(2, )[1],
        #             yerr=data_tmp.loc[:, data_tmp.columns.str.endswith("sd")].to_numpy().reshape(2, )[1],
        #             ecolor="black")
        ax.errorbar(data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].columns.to_numpy(),
                    data_tmp.groupby(["task", "correctnum_{}bit".format(tdata.bit)]).mean().loc[:,
                    data_tmp.groupby(["task", "correctnum_{}bit".format(tdata.bit)]).mean().columns.str.endswith(
                        "mean")].to_numpy().T,
                    yerr=np.mean(data_tmp.loc[:, data_tmp.columns.str.endswith("sd")].to_numpy(), axis=0),
                    ecolor="blue")
        # ax.errorbar(data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].columns[1],
        #             data_tmp.groupby(["task", "correctnum_{}bit".format(tdata.bit)]).mean().loc[:,
        #             data_tmp.groupby(["task", "correctnum_{}bit".format(tdata.bit)]).mean().columns.str.endswith(
        #                 "mean")].to_numpy().reshape(2, )[1],
        #             yerr=data_tmp.loc[:, data_tmp.columns.str.endswith("sd")].to_numpy().reshape(2, )[1],
        #             ecolor="black")
        # mean
        ax.plot(data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].columns,
                data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].to_numpy().T,
                marker="o", color="black")
        # all average
        ax.plot(data_tmp.loc[:, data_tmp.columns.str.endswith("mean")].columns,
                data_tmp.groupby(["task", "correctnum_{}bit".format(tdata.bit)]).mean().loc[:,
                data_tmp.groupby(["task", "correctnum_{}bit".format(tdata.bit)]).mean().columns.str.endswith(
                    "mean")].to_numpy().T,
                marker="x", color="blue")
        # plt.show(block=True)
        plt.savefig(os.path.join(os.getcwd(), 'fig', 'pattern_ent',
                                 'pattern_ent_average_{}_correct{}.png'.format(group_info[0], group_info[1])))


def test_base30():
    # error: 2,3,7,11,13,17,18

    mice = [6, 7, 8, 11, 12, 13, 14, 17, 18, 19, 23, 24, 90, 92]
    tasks = ["All5_30", "Only5_50", "Not5_Other30"]

    # 21,22 All5_30, Only5_70, Not5_Other30
    # mice = [21, 22]
    # tasks = ["All5_30", "Only5_70", "Not5_Other30"]

    # Base_51317, Test_51317
    # mice = [28]
    # tasks = ["Base_51317", "Test_51317"]

    logpath = './'
    tdata = task_data(mice, tasks, logpath)

    #    graph_ins = graph(tdata, mice, tasks, logpath)
    return tdata, mice, tasks


# tdata_30, mice_30, tasks_30 = test_base30()
# view_averaged_prob_same_prev(tdata_30, mice_30, tasks_30)


def test_base50():
    # All5_50, Only5_50, Not5_Other50, Recall5_50
    mice = [27]
    tasks = ["All5_50", "Only5_50", "Not5_Other50", "Recall5_50", "Cup_91019"]

    logpath = './'
    tdata = task_data(mice, tasks, logpath)

    return tdata, mice, tasks


tdata_50, mice_50, tasks_50 = test_base50()
view_averaged_prob_same_prev(tdata_50, mice_50, tasks_50)


# view_averaged_prob_same_prev(tdata_50, mice_50, tasks_50)


# TODO 下記のtest_Only5_70()から変数を返してもらう形式だとtask_dataインスタンスがlocalにならない
def test_Only5_70():
    # All5_50, Only5_50, Not5_Other50, Recall5_50
    mice = [21, 22]
    tasks = ["All5_30", "Only5_70", "Not5_Other30"]

    logpath = './'
    tdata = task_data(mice, tasks, logpath)

    view_averaged_prob_same_prev(tdata, mice, tasks)
    view_summary(tdata, mice, tasks)

    return tdata, mice, tasks

# TODO 下記のスクラッチをscientific modeで記述・実行する最も良い方法は何か？
# tdata_o5_70, mice_o5_70, tasks_o5_70 = test_Only5_70()
# view_averaged_prob_same_prev(tdata_o5_70, mice_o5_70, tasks_o5_70)

# TODO python-analyze以外の残骸branchを全削除

# 動物心理タイトル:「マウスの5本腕バンディット課題におけるwin-stay lose-shiftの法則の検証」
# TODO 動物心理コンセプト決め（証明したい仮説）:「」
# TODO 動物心理コンセプト決め（実験方法と実験）:「」
# TODO 動物心理コンセプト決め（仮説の検証結果）:「」
# TODO 動物心理Abstract
# 動物心理 Background 1. 動物のWSLSは2選択においてしか検討されていない（中央でholdもしくは3択で高確率ホール1つvs低確率2つはあるが)。
# 動物心理 Background 2. 2択の場合、loseの時のshift先がwinであることがほぼ確実なので、shiftすることへのリスクを伴う探索的要素が乏しい
# 動物心理 Background 3. そこで本研究では、loseの後のshift先が多肢・低確率であるような状況で、win-stay lose-shiftが成立しているかを検証する
# TODO 動物心理グラフ Fig.1. A. Summary B. burst表示(部分) C. ALL5_30 時間帯別トライアル数(correct/incorrect/omissionの積み上げバーグラフ, 個体別と全平均)
# TODO 動物心理グラフ Fig.2. Response Rate　A.タスク毎 (Prism) B. 2D-hist (ALL)
# TODO 動物心理グラフ Fig.3. Reward Latency A.タスク毎 (Prism) B. 2D-hist (ALL)
# TODO 動物心理グラフ Fig.4. P(same choice) {correct/incorrect start} A. タスク毎(burst考慮無) B. タスク毎(burst考慮有) burst_len を可変に
# TODO 動物心理グラフ Fig.5. P(same choice) A. Only5_50への切り替え直後(前半50correct) vs 終了直前(後半50correct) B.Not5_Other30
# TODO 動物心理グラフ Fig.6. 2bit比較(n>10以上, 全タスク（３つ）で作成) (Prism)
# TODO 動物心理グラフ Table.1. 体重変化(base, before, after), タスク終了に要した時間{All5_30, Only5_30, Not5_Other30}
# TODO 動物心理 Result 1. [Win-Stay Lose-Shiftはあるか?] 等確率の場合、報酬獲得選択時の3ステップ目まで報酬獲得選択時と同じ選択肢を選ぶ確率が10step先と比較して有意に高かった
# TODO 動物心理 Result 2. [WSLSは報酬確率のばらつきによって影響を受けるか？] 報酬確率が極端に異なるOnly5_50, Not5_Other30でも同様 (ベースの変化を別途同定しておいて、正規化して比較する)
# TODO 動物心理 Result 3. [WSLSは報酬確率のパターンの変化によって影響を受けるか？] タスクが切り替わった直後とタスクに適応した後で異なるか？ → (たぶん異ならない)
# TODO 動物心理 Result 4. [WSLSの効果は連続報酬によって変化するか？] 00, 01, 10 vs 11 → 連続効果は有意ではなかった？
# TODO 動物心理 Discussion 1.
