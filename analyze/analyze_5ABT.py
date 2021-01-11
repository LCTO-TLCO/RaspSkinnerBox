#
import pandas as pd
import numpy as np
from pyentrp import entropy as pent
from typing import Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.markers as markers
import seaborn as sns
import os
import sys
import csv
import time
from datetime import datetime, timedelta
import warnings
from skopt import gp_minimize
from skopt.plots import plot_convergence
from policy_func import *
from lerning_rule_func import *

verbose_level = 0
data_cache = {}
sns.set()
plt.style.use('ggplot')
NUM_SIM = 100

num_moving_average = 100  # 移動平均のstep数
num_first_to_remove = 100  # 移動平均にする際に削除するstep数

def get_model_list():
    models_dict = [
        # {"model_name": "DLR_Q_softmax_beta_free",
        #  "update_q_func": alpha_nega_posi_TD_error_DFQ_update,
        #  "policy_func": softmax_p,
        #  "pbounds": {"alpha_1": (0.000001, 0.3), "alpha_2": (0.000001, 0.3), "alpha_3": [0.0], "kappa_1": [1.0], "kappa_2": [0.0], "beta": (0.5, 20.0)}},
        # {"model_name": "standard_Q_softmax_beta_free",
        # "update_q_func": Q_update,
        # "policy_func": softmax_p,
        # "pbounds": {"alpha_1": (0.00001, 0.3), "kappa_1": [1.0], "kappa_2": [0.0], "beta": (0.0, 20.0)}},
        {"model_name": "DLR_Q_softmax_beta_5",
         "update_q_func": alpha_nega_posi_TD_error_DFQ_update,
         "policy_func": softmax_p,
         "pbounds": {"alpha_1": (0.000001, 0.3), "alpha_2": (0.000001, 0.3), "alpha_3": [0.0], "kappa_1": [1.0], "kappa_2": [0.0], "beta": [5]}},
        # {"model_name": "standard_Q_softmax_beta_5",
        #  "update_q_func": Q_update,
        #  "policy_func": softmax_p,
        #  "pbounds": {"alpha_1": (0.00001, 0.3), "kappa_1": [1.0], "kappa_2": [0.0], "beta": [5]}},
    ]
    return models_dict


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
    # 実験の意図
    # 実験条件の宣言
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
                 "mice": [118, 132, 175, 195, 201, 205, 208, 209, 210, 214],
                 },
        "BKLT": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"],
                 "mice": [119, 136, 149, 177, 190, 202, 206, 216, 219, 222, 225],
                 },
        "BKKO_Only5": {"tasks_section": ["All5_30", "Only5_50"],
                 "mice": [118, 132, 175, 195, 201, 205, 208, 209, 210, 214],
                 },
        "BKLT_Only5": {"tasks_section": ["All5_30", "Only5_50"],
                 "mice": [119, 136, 149, 177, 190, 202, 206, 216, 219, 222, 225],
                 },
        "BKtest": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"],
                 "mice": [222],
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


# まとめてグラフの形式を変更できるように関数とした
# 引数のprob=Trueであるとグラフの上限と下限が1,0に固定される
# 引数のbase_wに値が入っているとbase_wの横線が引かれる
def my_graph_plot(array, section_line, title, xlabel, ylabel, dirname=None, is_prob=False, base_w=False, is_save=False,
                  is_show=False):
    mpl.rcdefaults()
    plt.style.use('seaborn-colorblind')

    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Meirio']

    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.3

    plt.rcParams["legend.markerscale"] = 1.5
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["legend.edgecolor"] = 'gray'

    plt.rcParams['figure.dpi'] = 300

    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    if array.ndim == 2:  # 腕の本数だけグラフがある時
        for i in range(len(array.T)):
            label_name = i  # 腕の番号 0, 1, 2, 3, 4
            # label_name = 1 + i * 2 # holeの番号 1, 3, 5, 7, 9
            plt.plot(array.T[i], label=label_name)
        plt.legend()
    else:  # グラフが1本だけの時
        plt.plot(array)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 区間に線引きを行う
    for i in range(len(section_line) - 1):
        if is_prob:  # 確率である時、maxを1、minを0にする
            plt.vlines(section_line[i], 0, 1, "gray", linestyles='dashed')
        else:
            if base_w:
                plt.hlines(base_w, 0, len(array), "green", linestyles='dashed')  # 緑線をbase_weightに引く
                plt.vlines(section_line[i], min(np.min(array), base_w), max(np.max(array), base_w), "blue",
                           linestyles='dashed')
            else:
                plt.vlines(section_line[i], np.min(array), np.max(array), "gray", linestyles='dashed')
    if is_save:
        plt.savefig(dirname + title + '.eps', bbox_inches='tight', pad_inches=0.05)
        plt.title(title)  # タイトル
        plt.savefig(dirname + title + '.png', bbox_inches='tight', pad_inches=0.05)

    if is_show:
        plt.show()
    else:
        plt.close()


# タスク区間のstep数とグラフに描画するための区間の境界線を調べる
def check_section_size(mice_data, tasks):
    section_step = []
    section_line = [0]
    for i in tasks:
        section_step.append(sum(mice_data["task"] == i))
        section_line.append(section_step[-1] + section_line[-1])
        print(i, "\t:", section_step[-1], "step")  # プリント
    section_line.pop(0)
    # section_line.pop(-1)
    return section_step, section_line


# 使いやすいように行動選択と報酬のデータを加工し抜き出す
def extract_act_and_rew(mice_data):
    total_timesteps = len(mice_data)
    actions = []
    rewards = []

    for i in range(total_timesteps):
        if mice_data.iloc[i].hole_no == "0":  # T0(タスクコールするだけで餌がもらえていて行動選択をしていないらしい)などの行動の例外
            actions.append(-1)
        else:
            actions.append((int(mice_data.iloc[i].hole_no) - 1) / 2)
        if mice_data.iloc[i].event_type == "reward":
            rewards.append(1)
        else:
            rewards.append(0)
    act_and_rew = np.array([actions, rewards], dtype="int").T
    return act_and_rew


# グラフ用の配列作成：選択回数の合計
def sum_action_count(act_and_rew):
    total_timesteps = len(act_and_rew)
    total_action = np.zeros([NUM_ACTION])
    total_action_array = np.zeros([total_timesteps, NUM_ACTION])

    for i in range(total_timesteps):
        action, reward = act_and_rew[i]
        if action != -1:  # 例外以外はカウント
            total_action[action] += 1
        total_action_array[i] += total_action
    return total_action_array


# グラフ用の配列作成：報酬の移動平均
def moving_average_reward(act_and_rew, num_moving_average, num_first_to_remove):
    total_timesteps = len(act_and_rew)
    step_reward = []
    average_reward_array = []
    for i in range(total_timesteps):
        action, reward = act_and_rew[i]
        step_reward.append(reward)
        average_reward_array.append(np.mean(step_reward[-num_moving_average:]))
    average_reward_array = np.array(average_reward_array)
    average_reward_array[:num_first_to_remove] = 0
    return average_reward_array


# グラフ用の配列作成：選択回数の移動平均
def moving_average_action(act_and_rew, num_moving_average, num_first_to_remove):
    total_timesteps = len(act_and_rew)
    step_action = [[] for i in range(NUM_ACTION)]
    average_action_array = [[] for i in range(NUM_ACTION)]
    for i in range(total_timesteps):
        action, reward = act_and_rew[i]
        for j in range(NUM_ACTION):
            if action == j:
                step_action[j].append(1)
            else:
                step_action[j].append(0)
            average_action_array[j].append(np.mean(step_action[j][-num_moving_average:]))
    average_action_array = np.array(average_action_array).T
    average_action_array[:num_first_to_remove] = 0
    return average_action_array


# グラフ用の配列作成：1stepに経過した秒数
def moving_average_time(mice_data, num_moving_average, num_first_to_remove):
    total_timesteps = len(mice_data)
    timestamps = mice_data["timestamps"]
    step_time = []
    average_time_array = []
    for i in range(total_timesteps - 1):
        sec = (timestamps.iloc[i + 1] - timestamps.iloc[i]).total_seconds()
        step_time.append(sec)
        average_time_array.append(np.mean(step_time[-num_moving_average:]))
    average_time_array = np.array(average_time_array)
    average_time_array[:num_first_to_remove] = 0
    return average_time_array


# グラフ用の配列作成：タスク中のマウスの体重の上げ下げを見る
def calculation_weight(mice_data, weight_dict):
    base_weight, dep_weight, end_weight = weight_dict["base"], weight_dict["dep"], weight_dict["end"]
    total_timesteps = len(mice_data)
    total_seconds = (
            mice_data["timestamps"][total_timesteps - 1] - mice_data["timestamps"][0]).total_seconds()  # 実験でかかった秒数
    total_reward = np.sum(mice_data["is_correct"])  # 実験で行われた給餌回数
    oneday_seconds = 24 * 60 * 60  # 1日の秒数
    decrease_one_second = (base_weight - dep_weight) / oneday_seconds  # 1秒に減る体重
    increase_one_feeding = (end_weight - dep_weight + decrease_one_second * total_seconds) / total_reward  # 1回の給餌で増える体重

    weight_array = []
    for i in range(total_timesteps):
        elapsed_seconds = (mice_data["timestamps"][i] - mice_data["timestamps"][0]).total_seconds()
        increase_weight = increase_one_feeding * np.sum(mice_data["is_correct"][:i])
        decrease_weight = decrease_one_second * elapsed_seconds
        weight_array.append(dep_weight + increase_weight - decrease_weight)
    return np.array(weight_array)


# グラフ用の配列作成：直近 seconds_to_examine 秒前に食べた餌の数
def calculation_num_fooding(mice_data, seconds_to_examine):
    total_timesteps = len(mice_data)
    elapsed_seconds_array = np.array([])
    num_fooding_array = []
    for i in range(total_timesteps):
        elapsed_seconds = (mice_data["timestamps"][i] - mice_data["timestamps"][0]).total_seconds()
        if mice_data["is_correct"][i] == 1:
            elapsed_seconds_array = np.append(elapsed_seconds_array, elapsed_seconds)
        num_fooding_array.append(
            len(elapsed_seconds_array[elapsed_seconds_array > elapsed_seconds - seconds_to_examine]))
    return np.array(num_fooding_array)


# fullで作成したグラフを実験タスクの範囲に切り取り
def convert_section(full_array, tasks_section_dict, full_tasks_section_dict, full_section_line):
    fast_section = int(
        full_section_line[np.where(np.array(full_tasks_section_dict) == tasks_section_dict[0])[0][0] - 1])
    end_section = int(full_section_line[np.where(np.array(full_tasks_section_dict) == tasks_section_dict[-1])[0][0]])
    return np.array(full_array)[fast_section:end_section]


# 全区間での区間名を出してくれる
def create_full_tasks_section(mice_id, logpath):
    header = ["timestamps", "task", "session_id", "correct_times", "event_type", "hole_no"]
    csv_data = pd.read_csv("{}no{:03d}_action.csv".format(logpath, mice_id), names=header)
    task_data = csv_data["task"]

    tasks_section = ["None"]
    for i in task_data:
        if i != tasks_section[-1]:
            tasks_section.append(i)
    tasks_section.pop(0)
    return tasks_section


class Agent():
    def __init__(self, average_action_array, act_and_rew, section_line, num_first_to_remove=100):
        self.average_action_array = average_action_array
        self.act_and_rew = act_and_rew
        self.total_timesteps = len(act_and_rew)
        self.num_first_to_remove = num_first_to_remove
        self.other_dict = {"n": np.zeros([NUM_ACTION]), "v_value": np.zeros([NUM_ACTION]), "c": np.zeros([NUM_ACTION])}
        self.section_line = section_line

    def reset_other_dict(self):
        self.other_dict = {"n": np.zeros([NUM_ACTION]), "v_value": np.zeros([NUM_ACTION]), "c": np.zeros([NUM_ACTION])}

    def set_keys(self, keys):
        self.keys = keys
        self.num_params = len(keys)

    def set_other_dict(self, average_time_array, weight_array, weight_percent_array, num_fooding_array):
        self.other_dict["average_time"] = average_time_array
        self.other_dict["weight"] = weight_array
        self.other_dict["weight_percent"] = weight_percent_array
        self.other_dict["num_fooding"] = num_fooding_array

    def set_func(self, update_q_func, policy_func):
        self.update_q_func = update_q_func
        self.policy_func = policy_func

        try:
            self.parallel_update_q_func = eval("parallel_" + update_q_func.__name__)
            print("set parallel_update_q_func")
        except:
            print("not set parallel_update_q_func !!")
        try:
            self.parallel_policy_func = eval("parallel_" + policy_func.__name__)
            print("set parallel_policy_func")
        except:
            print("not set parallel_policy_func !!")


    def sim(self, params_dict):
        self.reset_other_dict()

        try:
            q_0_value = params_dict["q_0_value"]
        except:
            q_0_value = 0

        q_value = np.full([NUM_ACTION], q_0_value, dtype="float")
        self.other_dict["q_value_plus"] = np.full([NUM_ACTION], q_0_value, dtype="float")
        self.other_dict["q_value_minus"] =  np.full([NUM_ACTION], q_0_value, dtype="float")

        try:
            q_0_0_value = params_dict["q_0_0_value"]
            q_0_1_value = params_dict["q_0_1_value"]
            q_0_2_value = params_dict["q_0_2_value"]
            q_0_3_value = params_dict["q_0_3_value"]
            q_0_4_value = params_dict["q_0_4_value"]
            q_value = np.array([q_0_0_value, q_0_1_value, q_0_2_value, q_0_3_value, q_0_4_value])

        except:
            pass

        n = np.zeros([NUM_ACTION])
        q_value_array = np.zeros([self.total_timesteps, NUM_ACTION])
        prob_array = np.zeros([self.total_timesteps, NUM_ACTION])
        try:
            alpha_3 = params_dict["alpha_3"]
        except:
            alpha_3 = 0

        for i in range(self.total_timesteps):
            self.other_dict["i"] = i

            action, reward = self.act_and_rew[i]

            # self.other_dict["n"][action] = (self.other_dict["n"][action] + 1) / (1 - alpha_3)
            # self.other_dict["n"] = self.other_dict["n"] * (1 - alpha_3)

            prob = self.policy_func(q_value, params_dict, self.other_dict)  # 方策により確率の計算

            q_value_array[i] = q_value
            prob_array[i] = prob

            q_value = self.update_q_func(q_value, action, reward, params_dict, self.other_dict)  # 価値関数の更新

            ### 自己相関softmaxの設定更新
            action_one_hot = np.eye(NUM_ACTION)[action]
            try:
                self.other_dict["c"] = (1 - params_dict["tau"]) * self.other_dict["c"] + params_dict["tau"] * action_one_hot
            except:
                pass

        # prob_array[np.isnan(prob_array) == True] = 100
        # prob_array[np.isnan(prob_array) == True] = 1
        # prob_array[np.isnan(prob_array) == True] = 1e-3
        # prob_array = (prob_array.T / np.sum(prob_array, axis=1)).T
        return q_value_array, prob_array

    def call_sim(self, **params_dict):
        return self.sim(params_dict)

    def sim_f(self, params_dict):
        q_value_array, prob_array = self.sim(params_dict)
        error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        return np.sum(error_array)

    def call_sim_f(self, **params_dict):
        q_value_array, prob_array = self.sim(params_dict)
        error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        return np.sum(error_array)

    # set_keys で keys を設定してからでないと使えない
    def only_values_sim_f(self, values):
        params_dict = {}
        for i in range(self.num_params):
            params_dict[self.keys[i]] = values[i]

        #q_value_array, prob_array = self.sim(params_dict)

        # error を最小化
        #error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        #return np.sum(error_array)

        # ll を最小化
        # ll_array = calculate_ll(self.act_and_rew, prob_array, self.num_first_to_remove)
        # return -np.sum(ll_array)  # 最小化させたいので-をかける

        # unconstrained_sim の ll 最小化
        # q_value_array, prob_array = self.parallel_unconstrained_sim(params_dict, num_sim=1000)
        # ll_array = calculate_ll(self.act_and_rew, prob_array, self.num_first_to_remove)
        # return -np.sum(ll_array)

        # test
        q_value_array, prob_array = self.sim(params_dict)
        ll_array = calculate_ll(self.act_and_rew, prob_array, self.num_first_to_remove, self.section_line)

        ### each ll
        # unco_q_value_arrays, unco_prob_arrays = self.parallel_unconstrained_sim_return_arrays(params_dict, num_sim=NUM_SIM)
        #
        # unco_ll_arrays = parallel_calculate_ll(self.act_and_rew, unco_prob_arrays, self.num_first_to_remove)
        # unco_ll = np.mean(np.sum(unco_ll_arrays, axis=1))

        ### mean ll
        # mean_prob_array = np.mean(unco_prob_arrays, axis=0)
        # mean_unco_ll_array = calculate_ll(self.act_and_rew, mean_prob_array, self.num_first_to_remove)
        # mean_unco_ll = np.sum(mean_unco_ll_array)

        #w = 1
        return -np.sum(ll_array)
        #return -(np.sum(ll_array) + unco_ll)
        #return -(np.sum(ll_array) * w + unco_ll * (1 - w))
        #return -(np.sum(ll_array) + mean_unco_ll)
        #return -(np.sum(ll_array) + unco_ll + mean_unco_ll)

    # unconstrained_simをするための情報を入れる
    def set_unconstrained_sim_info(self, tasks_section, section_line, tasks_prob_dict):
        self.tasks_section = tasks_section
        self.section_line = section_line
        self.tasks_prob_dict = tasks_prob_dict

    def unconstrained_sim(self, params_dict):
        self.reset_other_dict()

        q_0_value = 0
        q_value = np.full([NUM_ACTION], q_0_value, dtype="float")
        n = np.zeros([NUM_ACTION])
        q_value_array = np.zeros([self.total_timesteps, NUM_ACTION])
        prob_array = np.zeros([self.total_timesteps, NUM_ACTION])
        try:
            alpha_3 = params_dict["alpha_3"]
        except:
            alpha_3 = 0

        for i in range(self.total_timesteps):
            self.other_dict["i"] = i
            prob = self.policy_func(q_value, params_dict, self.other_dict)

            # prob[np.isnan(prob) == True] = 1
            # prob = (prob / np.sum(prob))
            prob_array[i] = prob

            action = np.random.choice(NUM_ACTION, p=prob)  # 確率による腕の選択

            self.other_dict["n"][action] = (self.other_dict["n"][action] + 1) / (1 - alpha_3)
            self.other_dict["n"] = self.other_dict["n"] * (1 - alpha_3)

            if self.tasks_prob_dict[self.tasks_section[np.digitize(i, self.section_line)]][
                action] > np.random.random():  # タスク区間に従った確率で報酬を出す
                reward = 1
            else:
                reward = 0
            q_value = self.update_q_func(q_value, action, reward, params_dict, self.other_dict)  # 価値関数の更新
            q_value_array[i] = q_value

            ### 自己相関softmaxの設定更新
            action_one_hot = np.eye(NUM_ACTION)[action]
            try:
                self.other_dict["c"] = (1 - params_dict["tau"]) * self.other_dict["c"] + params_dict["tau"] * action_one_hot
            except:
                pass

        # prob_array[np.isnan(prob_array) == True] = 1
        #prob_array[np.isnan(prob_array) == True] = 1e-3
        # prob_array = (prob_array.T / np.sum(prob_array, axis=1)).T

        return q_value_array, prob_array



    # def parallel_unconstrained_sim(self, params_dict, num_sim=1000):
    #
    #     # FQ
    #     alpha_1 = params_dict["alpha_1"]
    #     alpha_2 = alpha_1
    #     alpha_3 = alpha_1
    #     kappa_1 = params_dict["kappa_1"]
    #     kappa_2 = params_dict["kappa_2"]
    #
    #     # DFQ
    #     # alpha_1 = params_dict["alpha_1"]
    #     # alpha_2 = alpha_1
    #     # alpha_3 = params_dict["alpha_2"]
    #     # kappa_1 = params_dict["kappa_1"]
    #     # kappa_2 = params_dict["kappa_2"]
    #
    #     # posi_nega_update_DFQ
    #     # alpha_1 = params_dict["alpha_1"]
    #     # alpha_2 = params_dict["alpha_2"]
    #     # alpha_3 = params_dict["alpha_3"]
    #     # kappa_1 = params_dict["kappa_1"]
    #     # kappa_2 = params_dict["kappa_2"]
    #
    #     q_0_value = 0
    #     beta = 1
    #
    #     p = self.tasks_prob_dict[self.tasks_section[0]]
    #     para_env = ParallelBanditEnv(p)
    #     para_agent = ParallelAgent(num_sim, alpha_1, alpha_2, alpha_3, kappa_1, kappa_2, q_0_value, beta)
    #
    #     q_value_array = np.zeros([self.total_timesteps, NUM_ACTION])
    #     prob_array = np.zeros([self.total_timesteps, NUM_ACTION])
    #
    #     for i in range(self.total_timesteps):
    #         para_env.change_p(self.tasks_prob_dict[self.tasks_section[np.digitize(i, self.section_line)]])
    #
    #         q_value_array[i] = np.mean(para_agent.q_values, axis=0)  # 記録
    #
    #         actions = para_agent.select_action()
    #         rewards = para_env.step(actions)
    #         para_agent.update(actions, rewards)
    #
    #         prob_array[i] = np.mean(para_agent.ps, axis=0)  # 記録
    #
    #     return q_value_array, prob_array


    def parallel_unconstrained_sim(self, params_dict, num_sim=1000):
        q_0_value = 0

        p = self.tasks_prob_dict[self.tasks_section[0]]
        para_env = ParallelBanditEnv(p)
        para_agent = ParallelAgent(num_sim, self.parallel_policy_func, self.parallel_update_q_func, q_0_value)

        q_value_array = np.zeros([self.total_timesteps, NUM_ACTION])
        prob_array = np.zeros([self.total_timesteps, NUM_ACTION])

        for i in range(self.total_timesteps):
            para_env.change_p(self.tasks_prob_dict[self.tasks_section[np.digitize(i, self.section_line)]])

            q_value_array[i] = np.mean(para_agent.q_values, axis=0)  # 記録

            actions = para_agent.select_action(params_dict, self.other_dict)
            rewards = para_env.step(actions)
            para_agent.update(actions, rewards, params_dict, self.other_dict)

            prob_array[i] = np.mean(para_agent.ps, axis=0)  # 記録

        return q_value_array, prob_array

    def parallel_unconstrained_sim_return_arrays(self, params_dict, num_sim=1000):
        q_0_value = 0

        p = self.tasks_prob_dict[self.tasks_section[0]]
        para_env = ParallelBanditEnv(p)
        para_agent = ParallelAgent(num_sim, self.parallel_policy_func, self.parallel_update_q_func, q_0_value)

        q_value_arrays = np.zeros([self.total_timesteps, num_sim, NUM_ACTION])
        prob_arrays = np.zeros([self.total_timesteps, num_sim, NUM_ACTION])

        for i in range(self.total_timesteps):
            para_env.change_p(self.tasks_prob_dict[self.tasks_section[np.digitize(i, self.section_line)]])

            q_value_arrays[i] = para_agent.q_values  # 記録

            actions = para_agent.select_action(params_dict, self.other_dict)
            rewards = para_env.step(actions)
            para_agent.update(actions, rewards, params_dict, self.other_dict)

            prob_arrays[i] = para_agent.ps # 記録

        q_value_arrays = q_value_arrays.transpose(1, 0, 2)
        prob_arrays = prob_arrays.transpose(1, 0, 2)

        return q_value_arrays, prob_arrays


    def call_mean_unconstrained_sim(self, **params_dict):
        return self.mean_unconstrained_sim(params_dict, num_sim=1)

    def mean_unconstrained_sim(self, params_dict, num_sim):

        # for文でぶん回す
        mean_q_value_array = np.zeros([self.total_timesteps, NUM_ACTION])
        mean_prob_array = np.zeros([self.total_timesteps, NUM_ACTION])
        for i in range(num_sim):
            q_value_array, prob_array = self.unconstrained_sim(params_dict)
            mean_q_value_array += q_value_array / num_sim
            mean_prob_array += prob_array / num_sim

        # parallelで回す
        # mean_q_value_array, mean_prob_array = self.parallel_unconstrained_sim(params_dict)

        # q_value_arrays, prob_arrays = self.parallel_unconstrained_sim_return_arrays(params_dict)
        # mean_q_value_array = np.mean(q_value_arrays, axis=0)
        # mean_prob_array = np.mean(prob_arrays, axis=0)

        return mean_q_value_array, mean_prob_array

    def mean_unconstrained_sim_return_arrays(self, params_dict, num_sim):

        # for文でぶん回す
        q_value_arrays = []
        prob_arrays = []
        for i in range(num_sim):
            q_value_array, prob_array = self.unconstrained_sim(params_dict)
            q_value_arrays.append(q_value_array)
            prob_arrays.append(prob_array)

        return np.array(q_value_arrays), np.array(prob_arrays)

    def mean_unconstrained_sim_f(self, params_dict, num_sim):
        q_value_array, prob_array = self.mean_unconstrained_sim(params_dict, num_sim)
        error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        return np.sum(error_array)

    def only_values_mean_unconstrained_sim_f(self, values):
        params_dict = {}
        for i in range(self.num_params):
            params_dict[self.keys[i]] = values[i]
        q_value_array, prob_array = self.mean_unconstrained_sim(params_dict, num_sim=100)
        error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        return np.sum(error_array)

    def unconstrained_sim_return_arrays(self, params_dict, num_sim):
        return self.parallel_unconstrained_sim_return_arrays(params_dict, num_sim)
        #return self.mean_unconstrained_sim_return_arrays(params_dict, num_sim)


def calculate_error(a, b, num_first_to_remove):  # 配列aとbとの自乗誤差の合計を返す
    error_array = (a - b) ** 2
    error_array[:num_first_to_remove] = 0  # num_first_to_remove stepまでのerrorは削除
    return error_array


def calculate_ll(act_and_rew, b, num_first_to_remove, section_line):
    # act_and_rew:マウスの選択と報酬履歴、b:強化学習による選択確率
    ll_array = np.zeros([len(b)])
    for i in range(len(b)):
        if num_first_to_remove < i:
            action, reward = act_and_rew[i]

            # 全ての腕に対してllを計算
            ll_array[i] = np.log(b[i][action])

            # 行動2のみに対してllを計算
            # if action == 2:
            #     ll_array[i] = np.log(b[i][action])

    # Otherを消すために
    # ll_array[:section_line[0]] = 0
    # ll_array[section_line[1]:] = 0

    # all区間のみ
    #ll_array[section_line[0]:] = 0

    # only区間のみ
    # ll_array[:section_line[0]] = 0
    # ll_array[section_line[1]:] = 0

    # Other区間のみ
    # ll_array[:section_line[1]] = 0

    return ll_array


def calculate_ll2(act_and_rew, b, num_first_to_remove):
    # act_and_rew:マウスの選択と報酬履歴、b:強化学習による選択確率
    ll_array = np.zeros([len(b)])
    for i in range(len(b)):
        if num_first_to_remove < i:
            action, reward = act_and_rew[i]
            if action == 2:
                ll_array[i] = np.log(b[i][2] + 1e-10)
            else:
                ll_array[i] = np.log(1 - b[i][2])

    return ll_array

def parallel_calculate_ll(act_and_rew, bs, num_first_to_remove):
    onehot_action = np.eye(NUM_ACTION)[np.array(act_and_rew).T[0]]
    ll_arrays = np.log(np.max(bs * onehot_action, axis=2))
    ll_arrays[:, :num_first_to_remove] = 0

    # choice_action = 2
    # one_action_onehot = np.eye(2)[(act_and_rew.T[0] == choice_action).astype(np.int)]
    # one_action_prob_array = np.stack([1 - bs.T[choice_action], bs.T[choice_action]], 0).T
    # ll_arrays = np.log(np.max(one_action_prob_array * one_action_onehot, axis=2))
    # ll_arrays[:, :num_first_to_remove] = 0

    return ll_arrays


# 区間に配列を切り分ける
def carve_out_section(array, section_line):
    carve_out_array = []
    for i in range(len(section_line)):
        if i == 0:
            carve_out_array.append(array[0:section_line[i]])
        else:
            carve_out_array.append(array[section_line[i - 1]:section_line[i]])
    return carve_out_array


class Optimisation():
    def __init__(self, agent, dirname, section_line, average_action_array, num_first_to_remove, act_and_rew):
        self.agent = agent
        self.model_results = {}
        self.model_order = []
        self.dirname = dirname + "bayesian_opt/"
        self.section_line = section_line
        self.average_action_array = average_action_array
        self.num_first_to_remove = num_first_to_remove
        self.act_and_rew = act_and_rew

    def do(self, mouse_id, update_q_func, policy_func, pbounds, n_calls, num_sim, is_show=False, is_save=False, model_name=None, is_required_unco_target=False, is_unco_check=True):
        values = [v for v in pbounds.values()]
        keys = [k for k in pbounds.keys()]
        self.agent.set_func(update_q_func, policy_func)
        self.agent.set_keys(keys)

        if model_name == None:
            model_name = update_q_func.__name__ + "+" + policy_func.__name__

        dirname = self.dirname + model_name + "/"
        os.makedirs(dirname, exist_ok=True)

        f = self.agent.only_values_sim_f
        # f = self.agent.only_values_mean_unconstrained_sim_f
        spaces = values
        start_time = time.time()
        last_time = time.time()

        params = []
        targets = []

        print("=================================================")

        #####################################################################
        for i in range(num_sim):
            res = gp_minimize(
                f, spaces,
                acq_func="gp_hedge",
                n_initial_points=round(n_calls * 0.3),
                n_calls=n_calls,
                acq_optimizer="lbfgs",
                n_jobs=-1,
                verbose=False)

            param = {}
            for j in range(len(keys)):
                param[keys[j]] = res["x"][j]
            params.append(param)
            targets.append(res["fun"])

            print(f"[id={mouse_id}][{model_name}][{i}] EV={res['fun']:.2f}, best:{param}, elapse={time.time() - last_time:.1f}s")
            last_time = time.time()

            q_value_array, prob_array = self.agent.sim(param)
            ll = np.sum(calculate_ll(self.act_and_rew, prob_array, self.num_first_to_remove, self.section_line))

            if (i + 1) % 5 == 0 and not((i + 1) == num_sim):
                print("--- on the way result -----------------------------")
                best_index = np.argmin(targets)
                best_params = params[best_index]

                print("best_params: ", best_params)
#                ll, error = self.save_sim(best_params, dirname, is_save=is_save, is_show=is_show)

                # if is_unco:
                #     unconstrained_sim_ll, unconstrained_sim_error = self.save_unconstrained_sim(best_params,
                #                                                                             dirname + "uncostrained_sim/",
                #                                                                             is_save=is_save,
                #                                                                             is_show=False,
                #                                                                             num_sim=NUM_SIM)
                # print("ll: ", ll)
                # print("error: ", error)
                #
                # ###
                # if is_unco:
                #     unco_q_value_arrays, unco_prob_arrays = self.agent.unconstrained_sim_return_arrays(best_params,
                #                                                                                                 num_sim=NUM_SIM)
                #     unco_ll_arrays = parallel_calculate_ll(self.act_and_rew, unco_prob_arrays, self.num_first_to_remove)
                #     unco_ll_array = np.mean(unco_ll_arrays, axis=0)
                #     unco_ll = np.sum(unco_ll_array)
                #     print("unconstrained_each_ll: ", unco_ll)
                #     print("unconstrained_mean_ll: ", unconstrained_sim_ll)
                #     print("ll + each ll: ", ll + unco_ll)
                #     print("unconstrained_sim_error: ", unconstrained_sim_error)

                print("end time: ", time.time() - start_time)

        print("--- total result -----------------------------")
        best_index = np.argmin(targets)
        best_params = params[best_index]
        ll, error = self.save_sim(best_params, dirname, is_save=is_save, is_show=is_show)

        print("best_params: ", best_params)
        print("ll: ", ll)
        print("error: ", error)

        if is_unco_check:
            unconstrained_sim_ll, unconstrained_sim_error = self.save_unconstrained_sim(best_params,
                                                                                    dirname + "uncostrained_sim/",
                                                                                    is_save=is_save, is_show=False,
                                                                                    num_sim=NUM_SIM)

            unco_q_value_arrays, unco_prob_arrays = self.agent.unconstrained_sim_return_arrays(best_params, num_sim=NUM_SIM)
            unco_ll_arrays = parallel_calculate_ll(self.act_and_rew, unco_prob_arrays, self.num_first_to_remove)
            unco_ll_array = np.mean(unco_ll_arrays, axis=0)
            unco_ll = np.sum(unco_ll_array)
            print("unconstrained_each_ll: ", unco_ll)
            print("unconstrained_mean_ll: ", unconstrained_sim_ll)
            print("ll + each ll: ", ll + unco_ll)
            print("unconstrained_sim_error: ", unconstrained_sim_error)

        print("end time: ", time.time() - start_time)

        for k, v in best_params.items():  # 範囲の端ならば注意メッセージを出す
            if any(np.array(pbounds[k]) == v) and not (len(pbounds[k]) == 1):
                print("* Note", k, ": pbounds =", pbounds[k], ", best_params =", v)

        result = best_params.copy()
        result["mouse_id"] = mouse_id
        result["ll"] = ll
        result["error"] = error
        result["model"] = model_name
        if is_unco_check:
            result["unconstrained_mean_ll"] = unconstrained_sim_ll
            result["unconstrained_sim_error"] = unconstrained_sim_error
            result["unconstrained_each_ll"] = unco_ll
            result["target"] = ll + unco_ll

        self.model_results[model_name] = result
        self.model_order.append(model_name)

        return result

    def save_model_results_csv(self):
        model_results = pd.DataFrame(self.model_results)
        model_results = model_results[self.model_order].T  # 実行した順に並び替える
        # model_results = model_results[["error", "unconstrained_sim_error", "alpha_1", "kappa_1", "kappa_2", "aleph", "a", "b", "epsilon"]].T
        model_results.to_csv(self.dirname + "model_results.csv")
        print("save model_results.csv")

    def save_sim(self, best_params, dirname, is_save=False, is_show=False):
        q_value_array, prob_array = self.agent.sim(best_params)
        ll_array = calculate_ll(self.act_and_rew, prob_array, self.num_first_to_remove, self.section_line)
        error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        my_graph_plot(q_value_array, self.section_line, "q_value_graph", "total timestep", "q value", dirname=dirname,
                      is_save=is_save, is_show=is_show)
        my_graph_plot(prob_array, self.section_line, "prob_graph", "total timestep", "prob", dirname=dirname,
                      is_save=is_save, is_show=is_show, is_prob=True)
        my_graph_plot(ll_array, self.section_line, "ll_graph", "total timestep", "ll", dirname=dirname, is_save=is_save,
                      is_show=is_show)
        my_graph_plot(error_array, self.section_line, "error_graph", "total timestep", "error", dirname=dirname,
                      is_save=is_save, is_show=is_show)
        return np.sum(ll_array), np.sum(error_array)

    def save_unconstrained_sim(self, best_params, dirname, is_save=False, is_show=False, num_sim=1000):
        os.makedirs(dirname, exist_ok=True)
        q_value_array, prob_array = self.agent.mean_unconstrained_sim(best_params, num_sim)
        ll_array = calculate_ll(self.act_and_rew, prob_array, self.num_first_to_remove, self.section_line)
        error_array = calculate_error(self.average_action_array, prob_array, self.num_first_to_remove)
        my_graph_plot(q_value_array, self.section_line, "q_value_graph", "total timestep", "q value", dirname=dirname,
                      is_save=is_save, is_show=is_show)
        my_graph_plot(prob_array, self.section_line, "prob_graph", "total timestep", "prob", dirname=dirname,
                      is_save=is_save, is_show=is_show, is_prob=True)
        my_graph_plot(ll_array, self.section_line, "ll_graph", "total timestep", "ll", dirname=dirname, is_save=is_save,
                      is_show=is_show)
        my_graph_plot(error_array, self.section_line, "error_graph", "total timestep", "error", dirname=dirname,
                      is_save=is_save, is_show=is_show)
        return np.sum(ll_array), np.sum(error_array)


class ParallelBanditEnv():
    def __init__(self, p):
        self.p = np.array(p)

    def change_p(self, p):
        self.p = np.array(p)

    def step(self, actions):
        reward = self.p[actions] > np.random.random(len(actions))
        return reward.astype(int)

class ParallelAgent():
    def __init__(self, num_sim, parallel_policy_func, parallel_update_q_func, q_0_value):
        self.num_sim = num_sim
        self.parallel_policy_func = parallel_policy_func
        self.parallel_update_q_func = parallel_update_q_func
        self.q_0_value = q_0_value
        self.q_values_reset()

    def q_values_reset(self):
        self.q_values = np.full((self.num_sim, NUM_ACTION), float(self.q_0_value))

    def select_action(self, params_dict, other_dict):
        self.ps = self.parallel_policy_func(self.q_values, params_dict, other_dict)
        x = self.ps.cumsum(axis=1) > np.random.rand(self.num_sim)[:, None]
        actions = x.argmax(axis=1)
        return actions

    def update(self, actions, rewards, params_dict, other_dict):
        self.q_values = self.parallel_update_q_func(self.q_values, actions, rewards, params_dict, other_dict)

def satori_main(mice_id, dirname=None):
    is_show = False  # グラフを表示するかしないか

    mice_id = mice_id  # 調べたいマウスのIDにする

    num_moving_average = 100  # 移動平均のstep数
    num_first_to_remove = 100  # 移動平均にする際に削除するstep数
    seconds_to_examine = 3600 * 12  # 直近食べた餌の数のグラフの参照する秒

    # mice_data_dict: 調べたいマウスの番号の情報を入力しておく必要あり
    #     --tasks_section: 分析したいマウスのタスク区間
    #     --full_tasks_section: 練習タスクを含めたマウスのタスク区間(餌を食べた量などを調べるときに使う)
    #     --weight: マウスの体重。不明ならば0を入力

    mice_data_dict = {
        "6": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "7": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "8": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "11": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "12": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "13": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 16.2, "dep": 15.0, "end": 16.2}},
        "14": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 16.6, "dep": 15.2, "end": 13.8}},
        "17": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 23.3, "dep": 20.1, "end": 22.6}},
        "18": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "21": {"tasks_section": ["All5_30", "Only5_70", "Not5_Other30"], "weight": {"base": 22.3, "dep": 20.3, "end": 17.0}},
        "22": {"tasks_section": ["All5_30", "Only5_70", "Not5_Other30"], "weight": {"base": 16.8, "dep": 13.9, "end": 17.0}},
        "23": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 21.4, "dep": 17.9, "end": 21.6}},
        "24": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 25.1, "dep": 23.2, "end": 20.4}},
        "26": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base":18.8, "dep": 16.1, "end": 17.0}},
        "27": {"tasks_section": ["All5_50", "Only5_50", "Not5_Other50", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "30": {"tasks_section": ["All5_50", "All5_30"], "weight": {"base": 22.8, "dep": 20.4, "end": 21.4}},
        "31": {"tasks_section": ["All5_50", "All5_30"], "weight": {"base": 22.1, "dep": 20.6, "end": 21.6}},
        "33": {"tasks_section": ["All5_50", "All5_30", "All5_10", "All5_10_drug"], "weight": {"base": 24.2, "dep": 20.5, "end": 18.2}},
        "34": {"tasks_section": ["All5_90", "All5_30", "All5_30_drug"], "weight": {"base": 16.3, "dep": 14.0, "end": 20.8}},
        "35": {"tasks_section": ["All5_90", "All5_30", "All5_30_drug"], "weight": {"base": 20.3, "dep": 19.8, "end": 21.2}},
        "36": {"tasks_section": ["All5_90", "All5_30", "All5_30_drug"], "weight": {"base": 21.6, "dep": 19.4, "end": 22.4}},
        "37": {"tasks_section": ["All5_90", "All5_30", "All5_30_drug"], "weight": {"base": 24.5, "dep": 20.7, "end": 22.5}},
        "38": {"tasks_section": ["All5_90", "All5_30"], "weight": {"base": 21.9, "dep": 19.6, "end": 21.8}},
        "41": {"tasks_section": ["All5_90"], "weight": {"base": 20.6, "dep": 19.8, "end": 20.5}},
        "42": {"tasks_section": ["All5_90", "All5_30", "All5_30_drug"], "weight": {"base": 21.5, "dep": 18.6, "end": 21.8}},
        "43": {"tasks_section": ["All5_90", "All5_30", "All5_30_drug"], "weight": {"base": 23.7, "dep": 20.1, "end": 22.4}},

        "47": {"tasks_section": ["All5_50", "All5_30", "All5_30_drug"], "weight": {"base": 17.6, "dep": 14.7, "end": 16.6}},
        "49": {"tasks_section": ["All5_50", "All5_30"], "weight": {"base": 19.4, "dep": 17.5, "end": 20.1}},

        "75": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 23.4, "dep": 20.2, "end": 18.9}},
        "76": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 25.4, "dep": 22.3, "end": 22.1}},

        "78": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 23.1, "dep": 20.7, "end": 20.6}},
        "79": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 27, "dep": 24.1, "end": 24.2}},
        "80": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 20, "dep": 16.7, "end": 20.3}},
        "81": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 23.5, "dep": 20.7, "end": 20.5}},

        "85": {"tasks_section": ["All5_60", "37_80", "159_60"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "88": {"tasks_section": ["All5_60", "Only5_50_plus30", "Not5_Other30_plus30"], "weight": {"base": 19.2, "dep": 17.7, "end": 19.3}},
        "90": {"tasks_section": ["All5_60", "Only5_50_plus30", "Not5_Other30_plus30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "92": {"tasks_section": ["All5_60", "Only5_50_plus30", "Not5_Other30_plus30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "94": {"tasks_section": ["All5_60", "Only5_50_plus30", "Not5_Other30_plus30"], "weight": {"base": 26.9, "dep": 26.5, "end": 26.5}},
        "95": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 20.4, "dep": 18.9, "end": 20.8}},

        "97": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 20.9, "dep": 18.3, "end": 21.3}},
        "98": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 22.7, "dep": 21.1, "end": 23.8}},

        "99": {"tasks_section": ["All5_20", "Only5_40", "Not5_Other20"], "weight": {"base": 19.5, "dep": 16.9, "end": 18.2}},
        "100": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 18.1, "dep": 16.4, "end": 17.9}},
        "101": {"tasks_section": ["All5_20", "Only5_40", "Not5_Other20"], "weight": {"base": 19.7, "dep": 18.7, "end": 18.1}},
        "102": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 0, "dep": 0, "end": 0}},

        "103": {"tasks_section": ["2arm5050", "2arm3070", "2arm7030"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "104": {"tasks_section": ["All5_20", "Only5_40", "Not5_Other20"], "weight": {"base": 21.9, "dep": 18.5, "end": 20.3}},
        "105": {"tasks_section": ["2arm5050", "2arm3070", "2arm7030"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "108": {"tasks_section": ["All5_45", "Only5_65", "Not5_Other45"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "110": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 18.8, "dep": 16.3, "end": 20.2}},
        "111": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 19.5, "dep": 17.9, "end": 21}},
        "112": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 16.9, "dep": 13.8, "end": 18.1}},
        "113": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 16.7, "dep": 14.3, "end": 17.3}},
        # "118": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        # "119": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "118": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "119": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "122": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 27.3, "dep": 26.5, "end": 24.4}},
        "128": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 27.0, "dep": 25.3, "end": 26.1}},
        "130": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 24.9, "dep": 22.0, "end": 23.8}},
        "132": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "136": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "137": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "138": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "141": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "144": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "148": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "149": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "152": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "154": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "155": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "160": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "163": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "165": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "168": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "169": {"tasks_section": ["All5_60", "Only5_80", "Not5_Other60"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "172": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "174": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "175": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"], "weight": {"base": 0, "dep": 0, "end": 0}},
        "177": {"tasks_section": ["All5_30", "Only5_50", "Not5_Other30", "Recall5_50"],
                "weight": {"base": 0, "dep": 0, "end": 0}},
    }

    tasks_prob_dict = {"All5_20": [0.2, 0.2, 0.2, 0.2, 0.2],
                       "All5_30": [0.3, 0.3, 0.3, 0.3, 0.3],
                       "All5_45": [0.45, 0.45, 0.45, 0.45, 0.45],
                       "All5_50": [0.5, 0.5, 0.5, 0.5, 0.5],
                       "All5_60": [0.6, 0.6, 0.6, 0.6, 0.6],
                       "All5_90": [0.9, 0.9, 0.9, 0.9, 0.9],
                       "Only5_40": [0, 0, 0.4, 0, 0],
                       "Only5_50": [0, 0, 0.5, 0, 0],
                       "Only5_65": [0, 0, 0.65, 0, 0],
                       "Only5_70": [0, 0, 0.7, 0, 0],
                       "Only5_80": [0, 0, 0.8, 0, 0],
                       "Not5_Other20": [0.2, 0.2, 0, 0.2, 0.2],
                       "Not5_Other30": [0.3, 0.3, 0, 0.3, 0.3],
                       "Not5_Other45": [0.45, 0.45, 0, 0.45, 0.45],
                       "Not5_Other50": [0.5, 0.5, 0, 0.5, 0.5],
                       "Not5_Other60": [0.6, 0.6, 0, 0.6, 0.6],
                       "Recall5_50": [0, 0, 0.5, 0, 0],
                       "All5_10_drug": [0.1, 0.1, 0.1, 0.1, 0.1],
                       "All5_30_drug": [0.3, 0.3, 0.3, 0.3, 0.3],
                       "37_80": [0, 0.8, 0, 0.8, 0],
                       "159_60": [0.6, 0, 0.6, 0, 0.6],
                       "Only5_50_plus30": [0.3, 0.3, 0.8, 0.3, 0.3],
                       "Not5_Other30_plus30": [0.6, 0.6, 0.3, 0.6, 0.6],

                       "2arm5050": [0.5, 0.5],
                       "2arm3070": [0.3, 0.7],
                       "2arm7030": [0.7, 0.3],
                       }

    logpath = './'  # 現在のディレクトリ直下

    mouse_group_dict = get_mouse_group_dict()
    choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
    mice = choice_mouse_group_dict["mice"]
    tasks = choice_mouse_group_dict["tasks_section"]

    choice_mice_data_dict = mice_data_dict[str(mice_id)]
    choice_mice_data_dict["full_tasks_section"] = create_full_tasks_section(mice_id, logpath)

    # 分析したいマウスのタスク区間のanalyzeによって加工されたデータの読み込み
    print("--- load section to analyze ---")
    tasks = choice_mice_data_dict["tasks_section"]
    task = analyze.task_data([mice_id], tasks, logpath)
    graph_ins = analyze.graph(task, [mice_id], tasks, logpath)
    mice_data = task.mice_task[mice_id]
    section_step, section_line = check_section_size(mice_data, tasks)
    act_and_rew = extract_act_and_rew(mice_data)

    # 全てのマウスのタスク区間のanalyzeによって加工されたデータの読み込み
    print("--- load full section ---")
    full_tasks = choice_mice_data_dict["full_tasks_section"]
    full_task = analyze.task_data([mice_id], full_tasks, logpath)
    full_graph_ins = analyze.graph(full_task, [mice_id], full_tasks, logpath)
    full_mice_data = full_task.mice_task[mice_id]
    full_section_step, full_section_line = check_section_size(full_mice_data, full_tasks)
    full_act_and_rew = extract_act_and_rew(full_mice_data)

    # タイムスタンプがstr型である時、timestamp型に変換
    if type(mice_data["timestamps"][0]) == type('string'):
        for i in range(len(mice_data["timestamps"])):
            mice_data["timestamps"][i] = pd.Timestamp(mice_data["timestamps"][i])
        for i in range(len(full_mice_data["timestamps"])):
            full_mice_data["timestamps"][i] = pd.Timestamp(full_mice_data["timestamps"][i])

    if all(np.array([i for i in choice_mice_data_dict["weight"].values()])) == 0:
        is_there_weight_info = False
    else:
        is_there_weight_info = True

    # 既にフォルダが有っても作成
    if  dirname == None:
        dirname = "mice_no" + str(mice_id) + "/"
    os.makedirs(dirname, exist_ok=True)

    # グラフ作成
    # ※エントロピーのグラフは宝田くんのプログラムで作成されたものを持ってきている。このグラフは150stepの移動平均エントロピーになっているよう。
    entropy_array = graph_ins.data.mice_task[mice_id]['hole_choice_entropy']
    total_action_array = sum_action_count(act_and_rew)
    average_reward_array = moving_average_reward(act_and_rew, num_moving_average, num_first_to_remove)
    average_action_array = moving_average_action(act_and_rew, num_moving_average, num_first_to_remove)
    average_time_array = moving_average_time(mice_data, num_moving_average, num_first_to_remove)

    # # 練習タスクでも餌を食べているのでそれを含めて計算し、通常タスクの区間に変換する
    full_weight_array = calculation_weight(full_mice_data, choice_mice_data_dict["weight"])
    full_num_fooding_array = calculation_num_fooding(full_mice_data, seconds_to_examine=seconds_to_examine)
    # 通常の区間へ変換
    weight_array = convert_section(full_weight_array, choice_mice_data_dict["tasks_section"],
                                   choice_mice_data_dict["full_tasks_section"], full_section_line)
    num_fooding_array = convert_section(full_num_fooding_array, choice_mice_data_dict["tasks_section"],
                                        choice_mice_data_dict["full_tasks_section"], full_section_line)

    # グラフの作成と保存
    my_graph_plot(entropy_array, section_line, "entropy_graph", "total timestep", "entropy", dirname, is_save=True,
                  is_show=is_show)
    my_graph_plot(total_action_array, section_line, "total_action_graph", "total timestep", "total action count",
                  dirname, is_save=True, is_show=is_show)
    my_graph_plot(average_reward_array, section_line, "moving_average_reward_graph", "total timestep",
                  "moving average reward", dirname, is_save=True, is_show=is_show)
    my_graph_plot(average_action_array, section_line, "moving_average_action_prob_graph", "total timestep",
                  "moving average action prob", dirname, is_prob=True, is_save=True, is_show=is_show)
    my_graph_plot(average_time_array, section_line, "moving_average_time_graph", "total timestep",
                  "moving average time [s]", dirname, is_save=True, is_show=is_show)
    my_graph_plot(weight_array, section_line, "weight_graph", "total timestep", "weight", dirname,
                  base_w=choice_mice_data_dict["weight"]["base"], is_save=True, is_show=is_show)
    my_graph_plot(num_fooding_array, section_line, "num_fooding_graph", "total timestep", "num fooding", dirname,
                  is_save=True, is_show=is_show)

    # 練習タスクも含めた全区間の確認用
    full_is_show = False
    full_tasls_dirname = dirname + "full_tasks/"
    os.makedirs(full_tasls_dirname, exist_ok=True)

    full_entropy_array = full_graph_ins.data.mice_task[mice_id]['hole_choice_entropy']
    full_total_action_array = sum_action_count(full_act_and_rew)
    full_average_reward_array = moving_average_reward(full_act_and_rew, num_moving_average, num_first_to_remove)
    full_average_action_array = moving_average_action(full_act_and_rew, num_moving_average, num_first_to_remove)
    full_average_time_array = moving_average_time(full_mice_data, num_moving_average, num_first_to_remove)

    my_graph_plot(full_entropy_array, full_section_line, "entropy_graph", "total timestep", "entropy",
                  full_tasls_dirname, is_save=True, is_show=full_is_show)
    my_graph_plot(full_total_action_array, full_section_line, "total_action_graph", "total timestep",
                  "total action count", full_tasls_dirname, is_save=True, is_show=full_is_show)
    my_graph_plot(full_average_reward_array, full_section_line, "moving_average_reward_graph", "total timestep",
                  "moving average reward", full_tasls_dirname, is_save=True, is_show=full_is_show)
    my_graph_plot(full_average_action_array, full_section_line, "moving_average_action_prob_graph", "total timestep",
                  "moving average action prob", full_tasls_dirname, is_prob=True, is_save=True, is_show=full_is_show)
    my_graph_plot(full_average_time_array, full_section_line, "moving_average_time_graph", "total timestep",
                  "moving average time [s]", full_tasls_dirname, is_save=True, is_show=full_is_show)
    my_graph_plot(full_weight_array, full_section_line, "weight_graph", "total timestep", "weight", full_tasls_dirname,
                  base_w=choice_mice_data_dict["weight"]["base"], is_save=True, is_show=full_is_show)
    my_graph_plot(full_num_fooding_array, full_section_line, "num_fooding_graph", "total timestep", "num fooding",
                  full_tasls_dirname, is_save=True, is_show=full_is_show)


def estimate_learning_rate_and_beta_importancesampling(mice, tasks, model):

    num_moving_average = 100  # 移動平均のstep数
    num_first_to_remove = 100  # 移動平均にする際に削除するstep数
    dirname = './'
    is_show = False

    for mouse_id in mice:
        start_time = time.time()
        # TODO : 樋口君

def estimate_learning_rate_and_beta(mice, tasks, model, num_sim=20, n_calls = 150):

    num_moving_average = 100  # 移動平均のstep数
    num_first_to_remove = 100  # 移動平均にする際に削除するstep数
    dirname = './'
    is_show = False

    for mouse_id in mice:
        start_time = time.time()

        print(f"[{mouse_id}] estimating learning rates")

        data = get_data(mouse_id)
        data = data[data["event_type"].isin(["reward", "failure"])]
        data = data[data["task"].isin(tasks)]
        data.reset_index()

        section_step, section_line = check_section_size(data, tasks)
        act_and_rew = extract_act_and_rew(data)

        average_action_array = moving_average_action(act_and_rew, num_moving_average, num_first_to_remove)
        average_time_array = moving_average_time(data, num_moving_average, num_first_to_remove)

        agent = Agent(average_action_array, act_and_rew, section_line, num_first_to_remove)
        agent.set_unconstrained_sim_info(tasks, section_line, get_tasks_prob_dict())
        opt = Optimisation(agent, dirname, section_line, average_action_array, num_first_to_remove, act_and_rew)

        ############## メイン実験
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

            result = opt.do(mouse_id=mouse_id,
                            update_q_func=model["update_q_func"],
                            policy_func=model["policy_func"],
                            pbounds=model["pbounds"],
                            n_calls=n_calls,
                            num_sim=num_sim,
                            is_show=is_show,
                            is_save=True,
                            is_required_unco_target=False,
                            is_unco_check=True,
                            model_name=model["model_name"])

            if 'df_estimation' in locals():
                df_estimation = df_estimation.append(pd.DataFrame.from_dict(result, orient='index').T)
            else:
                df_estimation = pd.DataFrame.from_dict(result, orient='index').T

    # TODO 宝田君 列名をmouse_id, model, ll, unconstrained_mean_ll・・・の順番にしたい
    # TODO ・・・以下はモデルによって不定
    # df_estimation = df_estimation.reindex(columns=['mouse_id', 'model', 'll', 'unconstrained_mean_ll']) #これでは消えてしまう

    print("total end time: ", time.time() - start_time)

    return df_estimation

def make_mouse_dir(mouse_id):
    dirname = f"no{mouse_id:03d}/"
    os.makedirs(dirname, exist_ok=True)
    return dirname

def plot_choiceratio_movingaverage_trialbase(mouse_id, tasks, mouse_group_name):
    """
    指定マウス1匹分の選択確率の移動平均を試行ステップで描画しeps, png保存
    :param mouse_id
    :return: なし
    """

#    dirname = make_mouse_dir(mouse_id)
    dirname = f"fig/{mouse_group_name}/"
    os.makedirs(dirname, exist_ok=True)

    data = get_data(mouse_id)
    data = data[data["event_type"].isin(["reward", "failure"])]
    data = data[data["task"].isin(tasks)]
    data.reset_index()

    section_step, section_line = check_section_size(data, tasks)
    act_and_rew = extract_act_and_rew(data)

    # グラフ作成
    average_action_array = moving_average_action(act_and_rew, num_moving_average, num_first_to_remove)
    average_reward_array = moving_average_reward(act_and_rew, num_moving_average, num_first_to_remove)

    # グラフの作成と保存
    my_graph_plot(average_action_array, section_line, f"Action probability {mouse_id} {mouse_group_name}", "Timestep (trial)",
                  "Averaged selection ratio", dirname, is_prob=True, is_save=True, is_show=True)
    my_graph_plot(average_reward_array, section_line, f"Reward probability {mouse_id} {mouse_group_name}", "Timestep (trial)",
                  "Averaged reward ratio", dirname, is_prob=True, is_save=True, is_show=True)


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
        # avg_c = np.append(avg_c, np.array([np.sum(np.array(prob_co),axis=0)/(len(correct_data)-b_min)]),axis=0)
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


# check TODO tdataの代わりにmouse_idで都度読み出しに変更 宝田君
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
    tdata = pd.DataFrame(
        columns=["timestamps", "mouse_id", "task", "session_id", "correct_times", "event_type", "hole_no"])
    for mouse_id in mice:
        tdata = pd.concat(
            [tdata, get_data(mouse_id).assign(mouse_id=mouse_id)])  # check mouse_idが無いので_export_P_20_filterでひっかかる
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
        trials = data[
            (data.task.isin([task])) & (data.event_type.isin(["reward", "failure"]))].hole_no.reset_index().hole_no
        # 該当するデータが存在しないときは中断して次のマウス
        if trials.size == 0:
            continue
        # 各関数の宣言
        conditions_function = (lambda x: x > threshold_ratio) if is_over else (lambda x: x < threshold_ratio)
        calc_select_prob_function = lambda d: d[d.isin([correct_hole])].size / window
        # 選択率の算出
        selection_raito = trials.rolling(window).apply(calc_select_prob_function, raw=False)
        # 選択率と閾値の比較
        result = selection_raito.apply(conditions_function)
        # 最初に選択率の条件を満たしたindexを取得
        first_trial = min(result[result].index) if not result[result].empty else -1
        # 結果を保存
        df_summary = df_summary.append({'mouse_id': mouse_id, 'trials': first_trial}, ignore_index=True)

        # BK KOマウスは、Only5_50で、hole 5のみが正解であることに気づくまで試行回数がかかることを定量化したい
        # BK KOマウスは、Not5_Other30で、hole 5に固執し、hole 5から脱却するまでの試行回数を算出したい

    return df_summary


def _get_session_id(mouse_id):
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
    data = data.session_id
    return data


def _rehash_session_id(data):
    session_id = 0
    print("max_id_col:{}".format(len(data)))

    def remove_terminate(index):
        if data.at[index, "event_type"] == data.at[index + 1, "event_type"] and data.at[
            index, "event_type"] == "start":
            data.drop(index, inplace=True)

    def rehash(x_index):
        nonlocal session_id
        if data.at[data.index[x_index], "task"] == "T0":
            if (x_index == 0 or data.shift(1).at[data.index[x_index], "event_type"] == "start") and \
                    len(data[:x_index][data.session_id == 0]) == 0:
                session_id = 0
                return 0
            session_id = session_id + 1
            return session_id
        if data.at[data.index[x_index], "event_type"] == "start":
            session_id = session_id + 1
            return session_id
        else:
            return session_id

    list(map(remove_terminate, data.index[:-1]))
    data.reset_index(drop=True, inplace=True)
    # 一回目
    data["session_id"] = list(map(rehash, data.index))
    data = data[data.session_id.isin(data.session_id[data.event_type.isin(["reward", "failure", "time over"])])]
    data.reset_index(drop=True, inplace=True)
    # 二回目
    session_id = 0
    data["session_id"] = list(map(rehash, data.index))
    return data


def calc_reactiontime_rewardlatency(mice):
    """
    マガジンノーズポークからhole選択までの反応時間(reactiontime)と正解後のマガジンノーズポークまでの時間(rewardlatency)をマウス・タスク毎に算出する
    :param mice: mouse_idのリスト
    :return: DataFrame, columns=['mouse_id', 'task', 'reaction_time', 'reward_latency'];　unique key=(mouse_id, task)
    """
    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'reaction_time', 'reward_latency'])

    for mouse_id in mice:
        data = get_data(mouse_id).assign(session_id=_get_session_id(mouse_id))
        data = _rehash_session_id(data)
        tasks_in_log = data.task.unique().tolist()
        deltas = {}
        for task in tasks_in_log:
            current_data = data[data.task == task]

            def calculate(session):
                delta_df = pd.DataFrame()
                # reaction time
                current_target = current_data[current_data.session_id.isin([session])]
                if bool(sum(current_target["event_type"].isin(["task called"]))) and bool(
                        sum(current_target["event_type"].isin(["nose poke"]))):
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
                    if pd.to_timedelta(reaction_time) / np.timedelta64(1, 's') > 500:
                        return None
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
                    if pd.to_timedelta(reward_latency) / np.timedelta64(1, 's') > 500:
                        return None
                    delta_df = delta_df.append(
                        {'type': 'reward_latency',
                         'noreward_duration_sec': pd.to_timedelta(norewarded_time) / np.timedelta64(1, 's'),
                         'reward_latency_sec': pd.to_timedelta(reward_latency) / np.timedelta64(1, 's')
                         }, ignore_index=True)
                if delta_df.size:
                    return delta_df
                else:
                    return None

            # mouse_id のtask 中ログから
            delta_df = current_data.session_id.drop_duplicates().map(calculate)
            tmp_df = None
            try:
                tmp_df = pd.concat(list(delta_df), sort=False)
            except:
                continue
            # 平均値を算出
            reaction_time = tmp_df[tmp_df.type.isin(["reaction_time"])].reaction_time_sec.mean()
            reward_latency = tmp_df[tmp_df.type.isin(["reward_latency"])].reward_latency_sec.mean()
            df_summary = df_summary.append(
                {"mouse_id": mouse_id, "task": task, "reaction_time": reaction_time, "reward_latency": reward_latency},
                ignore_index=True)

    return df_summary


# calculating entropy head 300/150 at each task
def calc_entropy(mice):
    df_summary = pd.DataFrame(columns=['mouse_id', 'task', 'entropy300', 'entropy150', 'entropy400'])

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
            ent400 = pent.shannon_entropy(hole_list.head(400).to_list())
            ent300 = pent.shannon_entropy(hole_list.head(300).to_list())
            ent150 = pent.shannon_entropy(hole_list.head(150).to_list())
            if verbose_level > 0:
                print(
                    f"[calc_entropy] mouse_id={mouse_id}, task={task: <13}: entropy300 = {ent300:.3f}, entropy150 = {ent150:.3f}, length={len(hole_list)}")

            ent_row = pd.Series([mouse_id, task, ent300, ent150, ent400], index=df_summary.columns)
            df_summary = df_summary.append(ent_row, ignore_index=True)

    return df_summary


def do_process(mouse_group_name):
    mouse_group_dict = get_mouse_group_dict()
    choice_mouse_group_dict = mouse_group_dict[mouse_group_name]
    mice = choice_mouse_group_dict["mice"]
    tasks = choice_mouse_group_dict["tasks_section"]

    is_plot = True
    is_calc_reaction_rewardlatency = True
    is_calc_entropy = True
    is_calc_stay_ratio = False
    is_calc_reach_threshold_ratio = True
    is_estimate_learning_params = True
    is_estimate_learning_params_importancesampling = False

    if is_plot:
        for mouse_id in mice:
            plot_choiceratio_movingaverage_trialbase(mouse_id, tasks, mouse_group_name)

    # reaction time & reward latency
    if is_calc_reaction_rewardlatency:
        df_rr = calc_reactiontime_rewardlatency(mice)
        print("[{}] Reaction time and Reward latency:".format(mouse_group_name))
        print(df_rr)
        df_rr.to_csv("./data/reaction_time/reaction_rewardlatency_{}.csv".format(mouse_group_name))


    # entropy計算
    if is_calc_entropy:
        df = calc_entropy(mice)
        df = df[df.task.isin(tasks)]

        # entropy 150 まとめて書き出し
        # dfp150 = df.pivot_table(index=['mouse_id'], columns=['task'], values=['entropy150'])  ## TODO: task実行順に並べたい
        # print(dfp150)
        # dfp150.to_csv("./data/entropy/entropy150_{}.csv".format(mouse_group_name))

        # entropy 300 まとめて書き出し
        dfp300 = df.pivot_table(index=['mouse_id'], columns=['task'], values=['entropy300'])  ## TODO: task実行順に並べたい
        dfp300 = dfp300.assign(group=mouse_group_name)
        print("[{}]entropy300:".format(mouse_group_name))
        print(dfp300)
        dfp300.to_csv("./data/entropy/entropy300_{}.csv".format(mouse_group_name))

        # # entropy 400 まとめて書き出し
        # dfp400 = df.pivot_table(index=['mouse_id'], columns=['task'], values=['entropy400'])  ## TODO: task実行順に並べたい
        # print(dfp400)
        # dfp400.to_csv("./data/entropy/entropy400_{}.csv".format(mouse_group_name))

    # WSLS
    if is_calc_stay_ratio:
        calc_stay_ratio(mice, tasks, selection=[1, 3, 5, 7, 9])

    #
    if is_calc_reach_threshold_ratio:
        # 正解選択率が50%を越えるまでのステップ数計算
        df_step_UP = calc_reach_threshold_ratio(mice, 'Only5_50', True, 0.5, 50, 5)
        df_step_UP = df_step_UP.assign(group=mouse_group_name)
        print("[{}]正解選択率が50%を越えるまでのステップ数:".format(mouse_group_name))
        print(df_step_UP)
        df_step_UP.to_csv("./data/step/step_UP50_{}.csv".format(mouse_group_name))

        # 非正解選択率が50%を下回るまでステップ数計算
        df_step_DOWN = calc_reach_threshold_ratio(mice, 'Not5_Other30', False, 0.5, 50, 5)
        df_step_DOWN = df_step_DOWN.assign(group=mouse_group_name)
        print("[{}]非正解選択率が50%を下回るまでステップ数計算:".format(mouse_group_name))
        print(df_step_DOWN)
        df_step_DOWN.to_csv("./data/step/step_DOWN50_{}.csv".format(mouse_group_name))

    if is_estimate_learning_params:
        models_dict = get_model_list()
        for model in models_dict:
            #df_learningparams = estimate_learning_rate_and_beta(mice, tasks, model, num_sim=30, n_calls=200)
            df_learningparams = estimate_learning_rate_and_beta(mice, tasks, model, num_sim=30, n_calls=250)
            df_learningparams = df_learningparams.assign(group=mouse_group_name)
            df_learningparams.to_csv("./data/estimation/params_{}_{}.csv".format(model["model_name"], mouse_group_name))

    if is_estimate_learning_params_importancesampling:
        models_dict = get_model_list()
        for model in models_dict:
            df_learningparams_is = estimate_learning_rate_and_beta_importancesampling(mice, tasks, model)
            #            df_learningparams = estimate_learning_rate_and_beta(mice, tasks, model, num_sim=15, n_calls=100)
            df_learningparams_is = df_learningparams_is.assign(group=mouse_group_name)
            df_learningparams_is.to_csv(
                "./data/estimation_is/params_{}_{}.csv".format(model["model_name"], mouse_group_name))

        print(df_learningparams)
    #return df_learningparams

do_process('BKKO')
do_process('BKLT')
do_process('BKKO_Only5')
do_process('BKLT_Only5')
#do_process('BKtest')

#df_lp_BKLT = do_process('BKLT')
#df_lp_BKKO = do_process('BKKO')
#df_learningparams = do_process('BKtest')

# if __name__ == '__main__':
#     args = sys.argv
#
#     ## 解析対象マウスの指定
#     if len(args) <= 1:
#         do_process('BKKO')
#         do_process('BKLT')
#     else:
#         do_process(args[1])
