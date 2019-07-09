import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd


class graph:
    def __init__(self, task_datas, mice, tasks, exportpath):
        plt.style.use("ggplot")
        font = {'family': 'meiryo'}
        mpl.rc('font', **font)
        self.data = task_datas
        self.mice = mice
        self.tasks = tasks
        self.exportpath = exportpath

    # data plot
    # TODO これは全nose pokeなので、burstは別に用意する
    # TODO window 2つ表示される
    def nose_poke_raster(self, mouse_id, ax):
        colors = ["blue", "red", "black"]
        labels = ["correct", "incorrect", "omission"]
        # flags = data.loc[:, data.colums.str.match("is_[(omission|correct|incorrect)")]
        datasets = [(self.data.mice_task[mouse_id][self.data.mice_task[mouse_id]
                                                   ["is_{}".format(flag)] == 1]) for flag in labels]
        for dt, la, cl in zip(datasets, labels, colors):
            ax.scatter(dt['session_id'] - dt['session_id'].min(), dt['is_hole1'] * 1, s=15, color=cl)
            ax.scatter(dt['session_id'] - dt['session_id'].min(), dt['is_hole3'] * 2, s=15, color=cl)
            ax.scatter(dt['session_id'] - dt['session_id'].min(), dt['is_hole5'] * 3, s=15, color=cl)
            ax.scatter(dt['session_id'] - dt['session_id'].min(), dt['is_hole7'] * 4, s=15, color=cl)
            ax.scatter(dt['session_id'] - dt['session_id'].min(), dt['is_hole9'] * 5, s=15, color=cl)
            ax.scatter(dt['session_id'] - dt['session_id'].min(), dt['is_omission'] * 0, s=15,
                       color=cl)
        ax.set_ylabel("Hole")
        plt.xlim(0, dt['session_id'].max() - dt['session_id'].min())

    def CFO_cumsum_plot(self, mouse_id, ax):
        ax.plot(self.data.mice_task[mouse_id]['cumsum_correct_taskreset'])
        ax.plot(self.data.mice_task[mouse_id]['cumsum_incorrect_taskreset'])
        ax.plot(self.data.mice_task[mouse_id]['cumsum_omission_taskreset'])
        plt.xlim(0, len(self.data.mice_task[mouse_id]))
        plt.ylabel('Cumulative')

    def entropy_scatter(self, mouse_id, ax):
        ax.plot(self.data.mice_task[mouse_id]['hole_choice_entropy'])
        plt.ylabel('Entropy (bit)')
        plt.xlim(0, len(self.data.mice_task[mouse_id]))

    def ent_raster_cumsum(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        for mouse_id in self.mice:
            self.entropy_scatter(mouse_id, fig.add_subplot(3, 1, 1))
            plt.title('{:03} summary'.format(mouse_id))
            self.nose_poke_raster(mouse_id, fig.add_subplot(3, 1, 2))
            self.CFO_cumsum_plot(mouse_id, fig.add_subplot(3, 1, 3))
            plt.xlabel('Trial')
            plt.show(block=True)
            plt.savefig('fig/{}no{:03d}_summary.png'.format(self.exportpath, mouse_id))

    def omission_plot(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        for mouse_id in self.mice:
            for task in self.tasks:
                # P(same) plot
                xlen = len(self.data.task_prob[mouse_id][task]["c_omit"])
                plt.subplot(1, len(self.tasks), self.tasks.index(task) + 1)
                plt.plot(self.data.task_prob[mouse_id][task]["f_omit"], label="incorrect")
                plt.plot(self.data.task_prob[mouse_id][task]["c_omit"], label="correct")
                plt.ion()
                plt.xticks(np.arange(1, xlen + 1, 1))
                plt.xlim(0.5, xlen + 0.5)
                plt.ylim(0, 1)
                if self.tasks.index(task) == 0:
                    plt.ylabel('P (omission)')
                    plt.legend()
                plt.xlabel('Trial')
                plt.title('{:03} {}'.format(mouse_id, task))
            plt.show(block=True)

            plt.savefig('fig/{}no{:03d}_omit.png'.format(self.exportpath, mouse_id))

    def same_plot(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        for mouse_id in self.mice:
            for task in self.tasks:
                # P(same) plot
                xlen = len(self.data.task_prob[mouse_id][task]["c_same"])
                plt.subplot(1, len(self.tasks), self.tasks.index(task) + 1)
                plt.plot(self.data.task_prob[mouse_id][task]["f_same"], label="incorrect")
                plt.plot(self.data.task_prob[mouse_id][task]["c_same"], label="correct")
                plt.ion()
                plt.xticks(np.arange(1, xlen + 1, 1))
                plt.xlim(0.5, xlen + 0.5)
                plt.ylim(0, 1)
                if self.tasks.index(task) == 0:
                    plt.ylabel('P (same choice)')
                    plt.legend()
                plt.xlabel('Trial')
                plt.title('{:03} {}'.format(mouse_id, task))
            plt.show(block=True)

            plt.savefig('fig/{}no{:03d}_prob.png'.format(self.exportpath, mouse_id))

    def reaction_scatter(self):
        fig = {}
        for mouse_id in self.mice:
            fig[mouse_id] = {}
            for task in self.tasks:
                fig[mouse_id][task] = plt.figure(figsize=(15, 8), dpi=100)
                data = self.data.mice_delta[mouse_id][task][
                    self.data.mice_delta[mouse_id][task].type == "reaction_time"]
                ax = fig[mouse_id][task].add_subplot(1, 1, 1)
                ax.scatter(
                    data[data["correct_incorrect"] == "correct"].noreward_duration_sec,
                    data[data["correct_incorrect"] == "correct"].reaction_time_sec,
                    label="correct")
                ax.scatter(
                    data[data["correct_incorrect"] == "incorrect"].noreward_duration_sec,
                    data[data["correct_incorrect"] == "incorrect"].reaction_time_sec,
                    label="incorrect")
                plt.title('{:03} reaction_time {}'.format(mouse_id, task))
                ax.set_xlabel("No reward duration (s)")
                ax.set_ylabel("Reaction time (s)")
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                plt.savefig('fig/{}no{:03d}_{}_reaction_time.png'.format(self.exportpath, mouse_id, task))
        plt.show(block=True)

    def reaction_hist2d(self):
        for mouse_id in self.mice:
            for task in self.tasks:
                if 'data_all' in locals():
                    data_all = data_all.append(self.data.mice_delta[mouse_id][task][
                                                   self.data.mice_delta[mouse_id][task].type == "reaction_time"])
                else:
                    data_all = self.data.mice_delta[mouse_id][task][
                        self.data.mice_delta[mouse_id][task].type == "reaction_time"]

        fig = plt.figure(figsize=(15, 8), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        h, xedges, yedges, img = ax.hist2d(data_all.noreward_duration_sec, data_all.reaction_time_sec,
                                           bins=[np.linspace(0, 1000, 51), np.linspace(0, 15, 31)])
        ax.grid()

        plt.title('reaction_time all mice all task')
        ax.set_xlabel("No reward duration (s)")
        ax.set_ylabel("Reaction time (s)")
        plt.savefig('fig/{}allmicetask_reaction_time_hist.png'.format(self.exportpath))
        plt.show(block=True)

    def norew_reward_latency_scatter(self):
        for mouse_id in self.mice:
            for task in self.tasks:
                fig = plt.figure(figsize=(15, 8), dpi=100)
                data = self.data.mice_delta[mouse_id][task][
                    self.data.mice_delta[mouse_id][task].type == "reward_latency"]
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(data.noreward_duration_sec, data.reward_latency_sec)
                plt.title('{:03} reward_latency {}'.format(mouse_id, task))
                ax.set_xlabel("No reward duration (s)")
                ax.set_ylabel("Reward latency (s)")
                plt.savefig('fig/{}no{:03d}_{}_reward_latency.png'.format(self.exportpath, mouse_id, task))
        plt.show(block=True)

    def norew_reward_latency_hist2d(self):
        # 各マウス 各タスク
        if False:
            for mouse_id in self.mice:
                for task in self.tasks:
                    fig = plt.figure(figsize=(15, 8), dpi=100)
                    data = self.data.mice_delta[mouse_id][task][
                        self.data.mice_delta[mouse_id][task].type == "reward_latency"]
                    ax = fig.add_subplot(1, 1, 1)
                    h, xedges, yedges, img = ax.hist2d(data.noreward_duration_sec, data.reward_latency_sec,
                                                       bins=[np.linspace(0, 1000, 51), np.linspace(0, 10, 21)])
                    ax.grid()
                    plt.title('{:03} reward_latency {}'.format(mouse_id, task))
                    ax.set_xlabel("No reward duration (s)")
                    ax.set_ylabel("Reward latency (s)")
                    plt.savefig('fig/{}no{:03d}_{}_reward_latency_hist2d.png'.format(self.exportpath, mouse_id, task))

        # 全マウス 各タスク TODO pandas依存の書き方に直す → histpandasにはないので不可
        for task in self.tasks:
            # data_all_mice = pd.DataFrame([], columns=data.columns)
            data_all_mice = pd.DataFrame([])
            for mouse_id in self.mice:
                data_all_mice = data_all_mice.append(self.data.mice_delta[mouse_id][task][
                                                         self.data.mice_delta[mouse_id][task].type == "reward_latency"])

            fig = plt.figure(figsize=(15, 8), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            h, xedges, yedges, img = ax.hist2d(data_all_mice.noreward_duration_sec, data_all_mice.reward_latency_sec,
                                               bins=[np.linspace(0, 1000, 51), np.linspace(0, 10, 21)])
            ax.grid()
            plt.title('All mice reward_latency {}'.format(task))
            ax.set_xlabel("No reward duration (s)")
            ax.set_ylabel("Reward latency (s)")
            plt.savefig('fig/{}allmice_{}_reward_latency_hist2d.png'.format(self.exportpath, task))

        # 全マウス 全タスク TODO pandas依存の書き方に直す
        fig = plt.figure(figsize=(15, 8), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        # data_all = pd.DataFrame([], columns=data.columns)
        data_all = pd.DataFrame([])
        for mouse_id in self.mice:
            for task in self.tasks:
                data_all = data_all.append(self.data.mice_delta[mouse_id][task][
                                               self.data.mice_delta[mouse_id][task].type == "reward_latency"])

        h, xedges, yedges, img = ax.hist2d(data_all.noreward_duration_sec, data_all.reward_latency_sec,
                                           bins=[np.linspace(0, 1000, 51), np.linspace(0, 10, 21)])
        ax.grid()
        plt.title('reward_latency ALL mice/task')
        ax.set_xlabel("No reward duration (s)")
        ax.set_ylabel("Reward latency (s)")
        plt.savefig('fig/{}all_micetask_reward_latency_hist2d.png'.format(self.exportpath))

        plt.show(block=True)

    # TODO ここから下

    def prob_same_base(self):
        """ fig1 """
        for mouse_id in self.mice:
            for task in self.tasks:
                data = self.data.fig_prob[mouse_id][task]["fig1"].drop("n")
                data = data.set_index(pd.Index(list(map(lambda x: x + 1, [int(num) for num in data.index]))))
                data.loc[0] = 0.2
                data = data.sort_index()
                ax = data.plot(title='{:03} prob_same_base_pattern {}'.format(mouse_id, task))
                # for pattern in list(data.columns):
                #     fig = plt.figure(figsize=(15, 8), dpi=100)
                #     fig_data = data[pattern]
                #     fig_data.plot(title='{:03} prob_same_base_pattern{} {}'.format(mouse_id, pattern, task))
                ax.set_xlabel("bit")
                ax.set_ylabel("P(same base)")
                ax.set_xlim(0, max([int(num) for num in data.index]))
                plt.gca().get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                plt.savefig('fig/{}no{:03d}_{}_fig1.png'.format(self.exportpath, mouse_id, task))
            plt.show(block=True)

    def prob_same_prev(self):
        """ fig2 """
        for mouse_id in self.mice:
            for task in self.tasks:
                data = self.data.fig_prob[mouse_id][task]["fig2"].drop("n")
                data = data.set_index(pd.Index(list(map(lambda x: x + 1, [int(num) for num in data.index]))))
                data.loc[0] = 0.2
                data = data.sort_index()
                ax = data.plot(title='{:03} prob_same_prev_pattern {}'.format(mouse_id, task))
                # for pattern in list(self.data.fig_prob[mouse_id][task]["fig2"].columns):
                # fig = plt.figure(figsize=(15, 8), dpi=100)
                # data = self.data.fig_prob[mouse_id][task]["fig2"][pattern]
                # data.plot(title='{:03} prob_same_prev_pattern{} {}'.format(mouse_id, pattern, task))
                ax.set_xlabel("bit")
                ax.set_ylabel("P(same before)")
                ax.set_xlim(0, max([int(num) for num in data.index]))
                # ここで軸目盛が消えてる
                # ax.set_xticks(np.array(
                # list(map(lambda x: x + 1,
                #          [int(num) for num in data.index])) # ))
                plt.gca().get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                plt.savefig('fig/{}no{:03d}_{}_fig2.png'.format(self.exportpath, mouse_id, task))
            plt.show(block=True)

    def prob_omit(self):
        for mouse_id in self.mice:
            for task in self.tasks:
                data = self.data.fig_prob[mouse_id][task]["fig3"].drop("n")
                data = data.set_index(pd.Index(list(map(lambda x: x + 1, [int(num) for num in data.index]))))
                data.loc[0] = 0.2
                data = data.sort_index()
                ax = data.plot(
                    title='{:03} prob_omit_pattern {}'.format(mouse_id, task))
                # for pattern in list(self.data.fig_prob[mouse_id][task]["fig3"].columns):
                # fig = plt.figure(figsize=(15, 8), dpi=100)
                # data = self.data.fig_prob[mouse_id][task]["fig3"][pattern]
                # data.plot(title='{:03} prob_omit_pattern{} {}'.format(mouse_id, pattern, task))
                ax.set_xlabel("bit")
                ax.set_ylabel("P(same base)")
                ax.set_xlim(0, max([int(num) for num in data.index]))
                plt.gca().get_xaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                plt.savefig('fig/{}no{:03d}_{}_fig3.png'.format(self.exportpath, mouse_id, task))
            plt.show(block=True)

    def next_10_ent(self):
        bit = self.data.bit
        pattern = range(0, pow(2, bit))
        for mouse_id in self.mice:
            for task in self.tasks:
                data = self.data.mice_task[mouse_id][self.data.mice_task[mouse_id].task == task]
                current_data = pd.DataFrame()
                for pat_tmp in pattern:
                    current_data["{:b}".format(pat_tmp).zfill(self.data.bit)] = data[
                        (data.pattern == pat_tmp) & (data.event_type.isin(["reward", "failure"]))].entropy_10.reset_index()

                ax = current_data.plot.hist()
                ax.set_xlabel("next 10 entropy")
                ax.set_ylabel("counts")
                ax.set_title('{:03} {} {}'.format(mouse_id, sys._getframe().f_code.co_name, task))
                plt.savefig(
                    'fig/{}no{:03d}_{}_{}.png'.format(self.exportpath, mouse_id, task, sys._getframe().f_code.co_name))
            plt.show(block=True)

    def norew_ent_10(self):
        """ hist2d """
        for mouse_id in self.mice:
            for task in self.tasks:
                fig = plt.figure(figsize=(15, 8), dpi=100)
                data = self.data.mice_delta[mouse_id][task][
                    self.data.mice_delta[mouse_id][task].type == "reward_latency"]
                ax = fig.add_subplot(1, 1, 1)
                H = ax.hist2d(data.noreward_duration_sec, data.reward_latency_sec,
                              bins=[np.linspace(0, 1000, 51), np.linspace(0, 15, 31)])
                plt.title('{:03} {} {}'.format(mouse_id, sys._getframe().f_code.co_name, task))
                ax.set_xlabel("No reward duration (s)")
                ax.set_ylabel("10 step entropy")
                fig.colorbar(H[3], ax=ax)
                plt.savefig(
                    'fig/{}no{:03d}_{} {}.png'.format(self.exportpath, mouse_id, sys._getframe().f_code.co_name, task))
            plt.show(block=True)

    def time_ent_10(self):
        for mouse_id in self.mice:
            for task in self.tasks:
                fig = plt.figure(figsize=(15, 8), dpi=100)
                data = self.data.mice_delta[mouse_id][task][
                    self.data.mice_delta[mouse_id][task].type == "reward_latency"]
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(data.noreward_duration_sec, data.reward_latency_sec)
                plt.title('{:03} {} {}'.format(mouse_id, sys._getframe().f_code.co_name, task))
                ax.set_xlabel("No reward duration (s)")
                ax.set_ylabel("10 step entropy")
                plt.savefig(
                    'fig/{}no{:03d}_{}_{}.png'.format(self.exportpath, mouse_id, task, sys._getframe().f_code.co_name))
        plt.show(block=True)

    def burst_raster(self):
        fig = plt.figure(figsize=(15, 8), dpi=100)
        labels = ["correct", "incorrect", "omission"]
        # flags = data.loc[:, data.colums.str.match("is_[(omission|correct|incorrect)")]
        for mouse_id in self.mice:

            datasets = [(self.data.mice_task[mouse_id][self.data.mice_task[mouse_id]
                                                       ["is_{}".format(flag)] == 1]) for flag in labels]
            ax = fig.add_subplot(1, 1, 1)
            for dt, la in zip(datasets, labels):
                ax.scatter(dt['session_id'], dt['is_hole1'] * 1, s=15, color="blue")
                ax.scatter(dt['session_id'], dt['is_hole3'] * 2, s=15, color="blue")
                ax.scatter(dt['session_id'], dt['is_hole5'] * 3, s=15, color="blue")
                ax.scatter(dt['session_id'], dt['is_hole7'] * 4, s=15, color="blue")
                ax.scatter(dt['session_id'], dt['is_hole9'] * 5, s=15, color="blue")
                ax.scatter(dt['session_id'], dt['is_omission'] * 0, s=15, color="red")
            ax.set_xlabel("time/sessions")
            ax.set_ylabel("hole dots")
            ax.ylim(-1, 6)
            plt.title('{:03} burst_raster'.format(mouse_id))
            plt.show(block=True)
            plt.savefig('fig/{}no{:03d}_reaction_time.png'.format(self.exportpath, mouse_id))
