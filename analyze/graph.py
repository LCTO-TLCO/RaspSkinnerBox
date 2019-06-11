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
                       color=cl)  # TODO omissionだけsession_idが重複している?
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

    def reaction_scatter(self):
        for mouse_id in self.mice:
            for task in self.tasks:
                fig = plt.figure(figsize=(15, 8), dpi=100)
                data = self.data.mice_delta[mouse_id][task][
                    self.data.mice_delta[mouse_id][task].type == "reaction_time"]
                fig.add_subplot(1, 1, 1).scatter(
                    pd.to_timedelta(data[data["correct_incorrect"] == "correct"].continuous_noreward_period),
                    pd.to_timedelta(data[data["correct_incorrect"] == "correct"].reaction_time))
                fig.add_subplot(1, 1, 1).scatter(
                    pd.to_timedelta(
                        data[data["correct_incorrect"] == "incorrect"].continuous_noreward_period),
                    pd.to_timedelta(data[data["correct_incorrect"] == "incorrect"].reaction_time))
                plt.title('{:03} reaction_time {}'.format(mouse_id, task))
                plt.legend()
                plt.savefig('fig/{}no{:03d}_reaction_time.png'.format(self.exportpath, mouse_id))
            plt.show(block=True)

    def reward_latency_scatter(self):
        for mouse_id in self.mice:
            for task in self.tasks:
                fig = plt.figure(figsize=(15, 8), dpi=100)
                data = self.data.mice_delta[mouse_id][task][
                    self.data.mice_delta[mouse_id][task].type == "reward_latency"]
                fig.add_subplot(1, 1, 1).scatter(pd.to_timedelta(data.continuous_noreward_period),
                                                 pd.to_timedelta(data.reward_latency))
                plt.title('{:03} reward_latency {}'.format(mouse_id, task))
                plt.legend()
                plt.savefig('fig/{}no{:03d}_{}_reward_latency.png'.format(self.exportpath, mouse_id, task))
            plt.show(block=True)