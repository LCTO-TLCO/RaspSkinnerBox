from datetime import datetime
import pandas as pd
import numpy as np
import math
from scipy.stats import entropy
from graph import graph

debug = False


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
        self.session_id = 0
        self.data_not_omission = None
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
            print("max_id_col:{}".format(len(data)))
            session_id = 0

            def rehash(x_index):
                if data.at[data.index[x_index], "task"] == "T0":
                    if data.at[data.index[x_index], "event_type"] == "start" or data.shift(1).at[
                        data.index[x_index], "event_type"] == "start":
                        return 0
                    self.session_id = self.session_id + 1
                    return self.session_id
                if data.at[data.index[x_index], "event_type"] == "start":
                    self.session_id = self.session_id + 1
                    return self.session_id
                else:
                    return self.session_id

            data["session_id"] = list(map(rehash, data.index))
            print("{} ; rehash done".format(datetime.now()))
            return data

        # TODO この関数が処理速度重い
        def add_timedelta():
            data = self.data
            data = data[data.session_id.isin(
                data[(data["event_type"] == "reward") | (data["event_type"] == "failure")]["session_id"])]
            deltas = {}
            for task in self.tasks:
                def calculate(session):
                    delta_df = pd.DataFrame()
                    # reaction time
                    current_target = data[data.session_id.isin(session)]
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
                    # TODO 区間entropy(未来方向に10step) for文の代わりにmap ヘルパー関数使用

                delta_df = data[data.task == task].groupby(["session_id"]).session_id.apply(calculate)
                deltas[task] = delta_df
            print("{} ; time delta added".format(datetime.now()))
            return deltas

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
            ent = [0] * 150
            for i in range(0, len(data[data.event_type.isin(['reward', 'failure'])]) - 150):
                denominator = 150.0  # sum([data["is_hole{}".format(str(hole_no))][i:i + 150].sum() for hole_no in range(1, 9 + 1, 2)])
                current_entropy = min_max(
                    [data["is_hole{}".format(str(hole_no))][i:i + 150].sum() / denominator for hole_no in
                     [1, 3, 5, 7, 9]])
                ent.append(entropy(current_entropy, base=2))
            # region Description
            data[data.event_type.isin(['reward', 'failure'])]["hole_choice_entropy"] = ent
            # endregion

            # burst
            # data["burst_group"] = 1
            # for i in range(1, len(data)):
            #     if data["timestamps"][i] - data["timestamps"][i - 1] <= datetime.timedelta(seconds=60):
            #         data["burst_group"][i] = data["burst_group"][i - 1]
            #         continue
            #     data["burst_group"][i] = data["burst_group"][i - 1] + 1
            print("{} ; hot vector added".format(datetime.now()))
            return data

        self.data = rehash_session_id()
        self.data_ci = self.data
        self.delta = add_timedelta()
        self.data = add_hot_vector()
        self.data_not_omission = self.data[
            self.data.session_id.isin(self.data.session_id[self.data.event_type.isin(["time over"])])]

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
            # calculate7
            probability["c_same"] = probability["c_same"] / after_c_all if not after_c_all == 0 else 0.0
            probability["c_diff"] = probability["c_diff"] / after_c_all if not after_c_all == 0 else 0.0
            probability["c_omit"] = probability["c_omit"] / after_c_all if not after_c_all == 0 else 0.0
            probability["c_checksum"] = probability["c_same"] + probability["c_diff"] + probability["c_omit"]
            probability["f_same"] = probability["f_same"] / after_f_all if not after_f_all == 0 else 0.0
            probability["f_diff"] = probability["f_diff"] / after_f_all if not after_f_all == 0 else 0.0
            probability["f_omit"] = probability["f_omit"] / after_f_all if not after_f_all == 0 else 0.0
            probability["f_checksum"] = probability["f_same"] + probability["f_diff"] + probability["f_omit"]

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

        # TODO
        def analyze_pattern(bit=4):
            pattern = {}
            pattern_range = range(0, pow(2, bit))
            for task in self.tasks:
                pattern[task] = {}
                prob = pd.DataFrame(columns=["{:04b}".format(i) for i in pattern_range],
                                    index=range(1, forward_trace)).fillna(0.0)

                data = self.data[self.data.task == task]
                # search pattern
                for single_pattern in pattern_range:
                    f_pattern_matching = lambda x: min([
                        (not np.isnan(data.shift(-(bit - i)).at[data.index[x], "is_correct"])) ==
                        bool(math.floor(single_pattern / pow(2, i)) % 2)
                        for i in range(0, bit)
                    ])
                    pattern[task].update({single_pattern: data[:-3][data[:-3].index.map(f_pattern_matching)]})
                # count
                f_same_base = lambda x, idx: data.at[data.index[x.index], "hole_no"] == data.shift(-idx).at[
                    data.index[x.index], "hole_no"]
                f_same_prev = lambda x, idx: data.shift(-idx + 1).at[data.index[x.index], "hole_no"] == \
                                             data.shift(-idx).at[data.index[x.index], "hole_no"]
                f_omit = lambda x, idx: bool(self.data_ci.iloc[x.index + idx].is_ommition)
                for pat_tmp in pattern.keys():
                    same_base = [pattern[task][pat_tmp].map(f_same_base, idx) for idx in range(1, bit)]
                    same_prev = [pattern[task][pat_tmp].map(f_same_prev, idx) for idx in range(1, bit)]
                    omit = [pattern[task][pat_tmp].map(f_omit, idx) for idx in range(1, bit)]
                    # TODO append

        def burst():
            # RFOで絞り込み
            # 時間を計算、バースト番号振り分け
            # データ全体の該当部分にはめ込む
            # 前の値で補完
            # df.fillna(method="ffill")
            pass

        count_all()
        count_task()
        analyze_pattern()
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
            reward_latency_data = self.mice_delta[mouse_no][task][
                self.mice_delta[mouse_no][task].type == "reward_latency"]
            reward_latency_data.to_csv('{}data/no{:03d}_{}_rewardlatency.csv'.format(self.logpath, mouse_no, task))
            self.task_prob[mouse_no][task].to_csv('{}data/no{:03d}_{}_prob.csv'.format(self.logpath, mouse_no, task))


# TODO 1. モデルクラス{ログからのQ値更新, 次ステップの行動選択予測, 予測との一致度の記録, 総合一致度の算出}
# TODO 2. 複数モデル×パラメータセット, パラメータセットの定義(GA,GP探索を見据えて), テストの実行(並列実行 joblib)
# TODO 3. hist2d 連続無報酬期間 vs 区間Entropy(10 step分) 全マウス・タスク毎
# TODO 4. 1111(正正正正)～0000(誤誤誤誤) fig1={P(基点とsame), N数}, fig2={P(一つ前とsame), N数}, fig3={P(omission)},fig4{次の10区間の区間エントロピー}, 4bit固定ではなくn bit対応で構築
# TODO 5. 横軸:時間（1時間単位） vs 縦軸:区間entropy(単位時間内), correct/incorrect/omission
# TODO 6. Burst raster plot

if __name__ == "__main__":
    print("{} ; started".format(datetime.now()))
    # mice = [6, 7, 8, 11, 12, 13, 17]
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
    # graph_ins.reaction_scatter()
    graph_ins.reaction_hist2d()
    # graph_ins.reward_latency_scatter()
    graph_ins.reward_latency_hist2d()

    print("{} ; all done".format(datetime.now()))
