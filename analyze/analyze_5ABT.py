#
import pandas as pd
import numpy as np
import sys
from pyentrp import entropy as pent
from datetime import datetime, timedelta
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.markers as markers
import os
import csv
import time
import seaborn as sns

verbose_level = 0
data_cache = {}


def get_tasks_prob_dict():
    tasks_prob_dict = {"All5_30": [0.3, 0.3, 0.3, 0.3, 0.3],
                       "All5_50": [0.5, 0.5, 0.5, 0.5, 0.5],
                       "All5_60": [0.6, 0.6, 0.6, 0.6, 0.6],
                       "Only5_50": [0, 0, 0.5, 0, 0],
                       "Only5_80": [0, 0, 0.8, 0, 0],
                       "Not5_Other30": [0.3, 0.3, 0, 0.3, 0.3],
                       "Not5_Other60": [0.6, 0.6, 0, 0.6, 0.6],
                       "Recall5_50": [0, 0, 0.5, 0, 0],
                       }
    return tasks_prob_dict


def get_mouse_group_dict():
    mouse_group_dict = {
        "Scarce": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"],
                   "mice": [8, 13, 17, 26, 137, 141, 144, 154, 168, 172],
                   },
        "Rich": {"tasks_section": ["All5_50", "Only5_80", "Not5_Other50"],
                 "mice": [95, 98, 100, 102, 110, 112, 113, 128, 130, 169],
                 },
        "All30": {"tasks_section": ["All5_30"],
                  "mice": [6, 11, 12, 14, 18, 19, 21, 23, 24, 123, 138, 146, 148, 152, 163, 174, 180, 192],
                  },
        "All50": {"tasks_section": ["All5_50"],
                  "mice": [27, 30, 31, 33, 47, 49, 50, 52, 64, 67, 68, 71, 116, 117, 181, 184, 186, 187],
                  },
        "BKKO": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"],
                 "mice": [118, 132, 175, 195, 201, 205],
                 },
        "BKLT": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"],
                 "mice": [119, 136, 149, 177, 190, 202, 206],
                 },
    }
    return mouse_group_dict


def show_task_summary(data):
    tasks = data.task.unique()
    event_times = pd.pivot_table(data[data.event_type.isin(["reward", "failure", "time over"])], index="event_type",
                                 columns="task", aggfunc="count").timestamps
    task_duration = pd.DataFrame(data.groupby("task").timestamps.max() - data.groupby("task").timestamps.min())
    task_duration.timestamps = task_duration.timestamps / np.timedelta64(1, 'h')
    ret_val = event_times.append(task_duration.T).fillna(0)
    ret_val = ret_val.rename(index={"timestamps": "duration in hours"})
    # 列の順番をtaskをやった順でソート
    ret_val = ret_val.loc[:, tasks]
    pd.options.display.precision = 1
    print(ret_val)


def load_from_action_csv(mouse_id):
    if verbose_level > 0:
        print(f"[load_from_action_csv] mouse_id ={mouse_id}")
    # 環境によって要変更
    # file = "./no{:03d}_action.csv".format(mouse_id)
    file = "./data/no{:03d}_action.csv".format(mouse_id)
    data = pd.read_csv(file, names=["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"],
                       parse_dates=[0])
    if isinstance(data.iloc[0].timestamps, str):
        data = pd.read_csv(file, parse_dates=[0])  # 何対策？ -> 一行目がカラム名だった場合の対策です
        data.columns = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
    data = data[["timestamps", "event_type", "task", "hole_no"]]
    return data


def get_data(mouse_id):
    global data_cache
    data = data_cache[mouse_id] if data_cache.get(mouse_id, False) else load_from_action_csv(mouse_id)
    return data


# TODO 佐鳥君
def estimate_learning_rate_and_beta(mice, models, tasks):
    df_estimation = pd.DataFrame(
        columns=['mouse_id', 'model', 'alpha1', 'alpha2', 'alpha3', 'beta', 'LL', 'AIC', 'unconstLL', 'unconstAIC'])
    for mouse_id in mice:
        print(f"[{mouse_id}] estimating learning rates")
        data = load_from_action_csv(mouse_id)

    return df_estimation


# TODO 最適df_estimationはmain側でファイル保存できるようにしておき、あればsimのときにそれをmodel_paramsとして使う形にする？

# TODO 1匹分, 横軸trial 佐鳥君
# def plot_choiceratio_movingaverage_trialbase(mouse_id):

# TODO n匹分, 横軸reward 佐鳥君
# def plot_choiceratio_movingaverage_rewardbase(mice):

# TODO 1匹分, 横軸trial 佐鳥君 (これは新規？)
# def plot_entropy_movingaverage_trialbase(mouse_id):

# TODO n匹分, 横軸reward 佐鳥君
# def plot_entropy_movingaverage_rewardbase(mice):

# TODO 1匹分, 横軸trial, (estimate_learning_rate_and_betaの結果1行をmodel_paramsとして指定) 佐鳥君
# 選択率とQ値
# def plot_constrained_simulation(mouse_id, model_params):

# TODO 1匹分, 横軸trial, (estimate_learning_rate_and_betaの結果1行をmodel_paramsとして指定) 佐鳥君
# 選択率とQ値
# def plot_unconstrained_simulation(mouse_id, model_params):


# TODO calc_stay_ratioに合わせて調整 宝田君
def _export_P_20_filter(dc, mice, all_sel_dc):
    b_min = 6
    b_max = 11
    #    hole_range = range(1,b_min)
    hole_range = range(1, b_max)
    avg_c = np.empty((0, len(hole_range)))
    avg_f = np.empty((0, len(hole_range)))
    avg_c_base = np.empty(0)
    avg_i_base = np.empty(0)

    for no in mice:
        # print("no:{}".format(no))
        mice_data = dc[(dc.mouse_id.isin([no]))]
        mice_allsel_data = all_sel_dc[all_sel_dc.mouse_id.isin([no])]
        prob_co = []
        prob_in = []
        prob_co_base = []
        prob_in_base = []
        correct_data = mice_data[mice_data.event_type.isin(["reward"])]
        incorrect_data = mice_data[mice_data.event_type.isin(["failure"])]
        print(f"no={no}, correct:n={len(correct_data)}, incorrect:n={len(incorrect_data)}")

        if len(mice_allsel_data) - b_max <= 0:
            print("empty!")
            continue

        # c_same
        # export: mice * hole prob
        skip = 0
        for idx, dat in correct_data.iterrows():
            # 後ろ10個を処理しない(10先がない)
            if idx > max(mice_allsel_data.index) - b_max:
                skip += 1
                continue
            tmp = np.zeros((len(hole_range),))
            for j in hole_range:
                if dat["hole_no"] == mice_allsel_data["hole_no"][idx + j]:
                    tmp[j - 1] += 1
            prob_co.append(np.copy(tmp))
        #        avg_c = np.append(avg_c, np.array([np.sum(np.array(prob_co),axis=0)/(len(correct_data)-b_min)]),axis=0)
        avg_c = np.append(avg_c, np.array([np.sum(np.array(prob_co), axis=0) / (len(correct_data) - skip)]), axis=0)

        # f_same
        # export: mice * hole prob
        skip = 0
        for idx, dat in incorrect_data.iterrows():
            tmp = np.zeros((len(hole_range),))
            if idx > max(mice_allsel_data.index) - b_max:
                skip += 1
                continue
            for j in hole_range:
                if dat["hole_no"] == mice_allsel_data["hole_no"][idx + j]:
                    tmp[j - 1] += 1
            prob_in.append(np.copy(tmp))
        #        avg_f = np.append(avg_f, np.array([np.sum(np.array(prob_in),axis=0)/(len(incorrect_data)-b_min)]),axis=0)
        avg_f = np.append(avg_f, np.array([np.sum(np.array(prob_in), axis=0) / (len(incorrect_data) - skip)]), axis=0)

        # base correctデータのみを対象， taskぶち抜き
        # export: mice
        for idx, dat in correct_data.iterrows():
            tmp = 0
            if idx > max(mice_allsel_data.index) - b_max:
                continue
            for j in range(b_min, b_max):
                if dat["hole_no"] == mice_allsel_data["hole_no"][idx + j]:
                    tmp += 1
            prob_co_base.append(tmp / (b_max - b_min))
        avg_c_base = np.append(avg_c_base, np.average(np.array(prob_co_base)))

        # incorrect base
        # export: mice
        for idx, dat in incorrect_data.iterrows():
            tmp = 0
            if idx > max(mice_allsel_data.index) - b_max:
                continue
            for j in range(b_min, b_max):
                if dat["hole_no"] == mice_allsel_data["hole_no"][idx + j]:
                    tmp += 1
            prob_in_base.append(tmp / (b_max - b_min))
        avg_i_base = np.append(avg_i_base, np.average(np.array(prob_in_base)))

    return {"c_same": avg_c, "f_same": avg_f, "c_same_base": avg_c_base, "f_same_base": avg_i_base}


# TODO tdataの代わりにmouse_idで都度読み出しに変更 宝田君
# TODO 冗長だけど、mouse_id毎に計算して、dataframeに統合する形の方がわかりやすかな？
# TODO CSV出力は呼び出し元で行う
# def calc_stay_ratio(tdata, mice, tasks, selection=[1, 3, 5, 7, 9]) -> dict:
def calc_stay_ratio(mice, tasks, selection=[1, 3, 5, 7, 9]) -> [dict, dict]:
    """
    指定タスクにおける特定のholeの平均選択率が指定した率を上回った、もしくは下回るまでのtrial数を返す
    :param mice: mouse_idのリスト
    :param tasks: 解析対象のtask
    :param selection: 解析対象とするhole no
    """
    # count_taskをclass task_dataの外で下記の仕様で再実装（classから消す必要はない）ßß

    if isinstance(selection, str):
        selection = [selection]
    elif isinstance(selection, int):
        selection = [str(selection)]
    selection = [str(num) for num in selection]
    tdata = pd.DataFrame(columns=["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"])
    for mouse_id in mice:
        tdata = pd.concat([tdata, get_data(mouse_id)]) # mouse_idが無いので_export_P_20_filterでひっかかる
    dc = tdata[tdata["event_type"].isin(["reward", "failure"]) & tdata.task.isin(tasks)]
    dc = dc.reset_index()

    tmp_dt = dc[dc["hole_no"].isin(selection)]
    prob = _export_P_20_filter(tmp_dt, mice, dc)
    prob["c_same"] = np.hstack((prob["c_same_base"].reshape((len(mice), 1)), prob["c_same"]))
    prob["f_same"] = np.hstack((prob["f_same_base"].reshape((len(mice), 1)), prob["f_same"]))

    print(f'tasks={tasks} c_same_base mean = {prob["c_same_base"].mean()}')
    print(f'tasks={tasks} f_same_base mean = {prob["f_same_base"].mean()}')

    correct_data = pd.DataFrame(prob["c_same"], index=mice)
    incorrect_data = pd.DataFrame(prob["f_same"], index=mice)

    # 3のタスク毎の任意の複数の選択肢毎の全マウス平均をcsv出力
    correct_data.to_csv("./data/WSLS/mice_task{}_hole{}_cstart.csv".format("-".join(tasks), "".join(selection)))
    incorrect_data.to_csv("./data/WSLS/mice_task{}_hole{}_fstart.csv".format("-".join(tasks), "".join(selection)))

    with open(f'./data/WSLS/mice_task{"-".join(tasks)}_hole{"".join(selection)}_serial.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        i = 0
        for idx in mice:
            for trial in range(0, 6):
                writer.writerow([idx, "-".join(tasks), trial, "win", prob["c_same"][i, trial]])
                writer.writerow([idx, "-".join(tasks), trial, "lose", prob["f_same"][i, trial]])
            i += 1

    return correct_data, incorrect_data


# count_task(tdata30, mice30, tasks30, [1, 3, 5, 7, 9])
# count_task(tdata50, mice50, tasks50, [1, 3, 5, 7, 9])

# TODO 宝田君
def calc_reach_threshold_ratio(mice, task, is_over, threshold_ratio, window, correct_hole):
    """
    指定タスクにおける特定のholeの平均選択率が指定した率を上回った、もしくは下回るまでのtrial数を返す
    :param mice: mouse_idのリスト
    :param task: 指定タスク(ひとつのみ)
    :param is_over: 上回る場合はTrue, 下回る場合はFalse
    :param threshold_ratio: 指定選択率
    :param window: 平均選択確率を計算するためのtime(trial) window
    :param correct_hole: 指定hole
    :return: DataFrame, columns=['mouse_id', 'trials']
    """
    df_summary = pd.DataFrame(columns=['mouse_id', 'trials'])

    for mouse_id in mice:
        data = get_data(mouse_id)
        tasks_in_log = data.task.unique().tolist()

        # 以下にコード
        # trials = 指定taskにおいて、
        # 各windowステップ内(task先頭+windowから)でのcorrect_hole選択率が
        # threshold_ratioを超えるまで(is_over = trueの場合), もしくは下回るまでis_over=falseの場合)の
        # reward/failureに基づく(timeoverは除く)trial数
        # データの抜出
        trials = data[(data.task.isin([task])) & (data.event_type.isin(["reward", "failure"]))].hole_no.reset_index().hole_no
        # 各関数の宣言
        conditions_function = (lambda x: x > threshold_ratio) if is_over else (lambda x: x < threshold_ratio)
        calc_select_prob_function = lambda d: d[d.isin([correct_hole])].size / window
        # 選択率の算出
        selection_raito = trials.rolling(window).apply(calc_select_prob_function)
        # 選択率と閾値の比較
        result = selection_raito.apply(conditions_function)
        # 最初に選択率の条件を満たしたindexを取得
        first_trial = min(result[result].index)
        # TODO ＠タカラダ　結果を保存


        # BK KOマウスは、Only5_50で、hole 5のみが正解であることに気づくまで試行回数がかかることを定量化したい
        # BK KOマウスは、Not5_Other30で、hole 5に固執し、hole 5から脱却するまでの試行回数を算出したい

    return df_summary


def calc_reactiontime_rewardlatency(mice):
    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'reaction_time', 'reward_latency'])

    for mouse_id in mice:
        data = get_data(mouse_id)
        tasks_in_log = data.task.unique().tolist()

    return df_summary


# calculating entropy head 300/150 at each task
def calc_entropy(mice):
    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'entropy300', 'entropy150'])

    for mouse_id in mice:
        data = get_data(mouse_id)
        tasks_in_log = data.task.unique().tolist()

        for task in tasks_in_log:
            tasks_data = data[data.task.isin([task])]
            tasks_data_p = pd.pivot_table(tasks_data[tasks_data.event_type.isin(["reward", "failure", "time over"])],
                                          index="event_type", columns="hole_no", aggfunc="count").timestamps.fillna(0)
            tasks_data_p.loc["total_trials"] = tasks_data_p.sum()
            if verbose_level > 1:
                print('')
                print(task)
                print(tasks_data_p)  ## TODO: 回数なので整数で表示したい

            hole_list = tasks_data[tasks_data.event_type.isin(["reward", "failure"])]["hole_no"]
            ent300 = pent.shannon_entropy(hole_list.head(300).to_list())
            ent150 = pent.shannon_entropy(hole_list.head(150).to_list())
            if verbose_level > 0:
                print(
                    f"[calc_entropy] mouse_id={mouse_id}, task={task: <13}: entropy300 = {ent300:.3f}, entropy150 = {ent150:.3f}, length={len(hole_list)}")

            ent_row = pd.Series([mouse_id, task, ent300, ent150], index=df_summary.columns)
            df_summary = df_summary.append(ent_row, ignore_index=True)

    return df_summary


if __name__ == '__main__':
    args = sys.argv

    ## 解析対象マウスの指定
    if len(args) <= 1:
        mouse_group_name = "BKKO"  # default
    else:
        mouse_group_name = args[1]

    mouse_group_dict = get_mouse_group_dict()
    choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
    mice = choice_mouse_group_dict["mice"]
    tasks = choice_mouse_group_dict["tasks_section"]

    if len(args) >= 3:  # 個別に指定があった場合、そのマウスのみ解析
        mice = [args[2]]

    ## entropy計算実行
    df = calc_entropy(mice)
    df = df[df.task.isin(tasks)]

    # entropy 150 まとめて書き出し
    dfp150 = df.pivot_table(index=['mouse_id'], columns=['task'], values=['entropy150'])  ## TODO: task実行順に並べたい
    print(dfp150)
    dfp150.to_csv("./data/entropy/entropy150_{}.csv".format("-".join([str(n) for n in mice])))

    # entropy 300 まとめて書き出し
    dfp300 = df.pivot_table(index=['mouse_id'], columns=['task'], values=['entropy300'])  ## TODO: task実行順に並べたい
    print(dfp300)
    dfp300.to_csv("./data/entropy/entropy300_{}.csv".format("-".join([str(n) for n in mice])))

    # WSLS
    calc_stay_ratio(mice, tasks, selection=[1, 3, 5, 7, 9])

    # # 正解選択率が50%を越えるまでのステップ数計算
    # df_step_UP = calc_reach_threshold_ratio(mice, 'Only5_50', True, 0.5, 10, 5)
    # print(df_step_UP)
    # df_step_UP.to_csv("./data/step/step_UP50_{}.csv".format("-".join([str(n) for n in mice])))
    #
    # # 非正解選択率が50%を下回るまでステップ数計算
    # df_step_DOWN = calc_reach_threshold_ratio(mice, 'Not5_Other30', False, 0.5, 10, 5)
    # print(df_step_DOWN)
    # df_step_DOWN.to_csv("./data/step/step_DOWN50_{}.csv".format("-".join([str(n) for n in mice])))

# TODO 考え方:
#  0.意識レベルの低下した人でも流れがわかるようにする
#  1.命令はシンプルに(構造体で渡さずmouse_idなどの数字で渡す)
#  2.各関数の中で必要な構造体は冗長でも都度getしに行く(キャッシュは使ってもよいが、他の関数で変更した構造体データに依存してはならない)
#  3.複数匹のデータは、予め各関数の冒頭で決めて置いた列を持ったDataFrameで返す
#  4.呼び出し元(main or jupyter)で、帰ってきたDataFrameの加工を行い、統計ソフト側で読めるように調整する(pivotの異なるcsvフォーマットが複数ありえる)
#  5.グラフ出力は各関数の中で完結し、eps, pngを吐き出す(epsにエラーが出るものはpdfで吐く)
#  6.呼び出し元で、条件毎にmiceを定義し、条件を指定してまとめて処理する
#  7.特定の関数の中からしか呼ばれない汎用性の無い関数名は _ から始める
#  8.mouse_group_dict["tasks_section"]は解析したい対象taskを書く。はじくのは呼び出し元(main)
