from random import random
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from src.datamanipulator import DataManipulator


def plot_frame_spike(name: str, spike_frame: pd.DataFrame, color: str) -> None:
    """
    This method plots the spikes
    :param str name: the title of the plot
    :param pd.DataFrame spike_frame: the dataframe
    :param str color: the color of the plot
    """
    plt.scatter(spike_frame["relative firing time"], spike_frame["IF"],
                c=color)
    plt.title(name)
    plt.xlabel("relative firing time")
    plt.ylabel("IF")
    plt.show()


class Plotter:
    """
    This class plots the data.
    """
    def __init__(self, data_class: DataManipulator):
        """
        :param DataManipulator data_class: the DataManipulator class
        """
        self.data_class = data_class
        self.data = data_class.data
        self.dict = data_class.dict
        self.names = data_class.names

    def plot_spike(self, name: str, spike_name: str, color: str, all: bool, chosen_cells: list) -> None:
        """
        This method plots relative firing time and IF of the given cell's spike.
        :param str name: the name of the cell
        :param str spike_name: the number of the spike, for example "1.spike"
        :param str color: the color of the plot
        :param bool all: if true the common dict from all the cells will be plotted, if false the given cell
        """
        if all:
            all_dict = self.data_class.all_in_one_dict
            plt.scatter(all_dict[spike_name]["relative firing time"],
                        all_dict[spike_name]["IF"], c=color)
            plt.ylim(0, 400)
        elif isinstance(chosen_cells, list):
            all_dict = self.data_class.all_in_one_dict_creating(gbz_dict=self.data_class.choose_cells, chosen_cells=chosen_cells)
            plt.scatter(all_dict[spike_name]["relative firing time"],
                        all_dict[spike_name]["IF"], c=color)
            plt.ylim(0, 400)
        else:
            plt.scatter(self.dict[name][spike_name]["relative firing time"],
                        self.dict[name][spike_name]["IF"], c=color)
            plt.ylim(0, 400)
        if all:
            plt.title("All spikes")
        else:
            plt.title(name)
        plt.xlabel("relative firing time")
        plt.ylabel("IF")
        plt.show()

    def plot_dict_spikes(self, cell_name: str) -> None:
        """
        Plots the spikes of the given dictionary.
        :param str cell_name: the name of the cell
        """
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(self.dict[cell_name]))]
        for i in range(1, len(self.dict[cell_name]) + 1):
            spike = f"{i}.spike"
            self.plot_spike(name=cell_name, spike_name=spike, color=color[i - 1], all=False)

    @staticmethod
    def plot_fitted_data(x, data, final, log, spike, plot_name, range_spike):
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(range_spike)]
        function_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(range_spike)]
        if log:
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(10 ** x, 10 ** data, 'o', c=colors[spike], label=f"{spike+1}.spike")
            plt.plot(10 ** np.linspace(np.min(x), np.max(x), 201),
                     10 ** final,
                     'r', c=colors[spike])
            plt.title(plot_name)
            plt.ylim(0, 400)
            plt.xlabel("relative firing time")
            plt.ylabel("IF")
        else:
            plt.plot(x, data, 'o', c=colors[spike], label=f"{spike+1}.spike")
            plt.plot(np.linspace(np.min(x), np.max(x), 201), final, 'r', c=colors[spike])
            plt.title(plot_name)
            plt.ylim(0, 400)
            plt.legend([f"{spike}.spike"])
            plt.xlabel("relative firing time")
            plt.ylabel("IF")
            # plt.show()

    @staticmethod
    def plot_fitted_data_eval(x, data, final, log, plot_name, ax, id):
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(1)]
        function_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(1)]
        if log:
            ax[id].set_xscale('log')
            ax[id].set_yscale('log')
            ax[id].plot(10 ** x, 10 ** data, 'o', c=colors[0])
            ax[id].plot(10 ** x, 10 ** final, 'r', c=colors[0])
            ax[id].set_title(plot_name)
            ax[id].set_xlabel("relative firing time")
            ax[id].set_ylabel("IF")
        else:
            ax[id].set_xlim(0, 280)
            ax[id].set_ylim(0, 340)
            ax[id].plot(x, data, 'o', c=colors[0])
            ax[id].plot(x, final, 'r', c=colors[0])
            ax[id].set_title(plot_name)
            ax[id].set_xlabel("relative firing time")
            ax[id].set_ylabel("IF")
            # plt.show()

    @staticmethod
    def different_if_plotter(df, p, ax, idx, threshold):
        ax[idx].set_xlim(0, np.log10(280))
        ax[idx].set_ylim(0, threshold[-1])
        ax[idx].scatter(df["relative firing time"], df["IF"])
        ax[idx].plot(df["relative firing time"], p[0] + p[1]*df["relative firing time"])
        ax[idx].set_title("Threshold= " + str(round(10 ** threshold[idx])))
        ax[idx].set_xlabel("relative firing time")
        ax[idx].set_ylabel("IF")

    @staticmethod
    def plot_errors(dictionary: dict, threshold: list, what_to_plot: str, cell_name, spike_name):
        arang = np.arange(1, len(threshold)+1, 1)
        for i, num in enumerate(threshold):
            plt.plot(arang[i], dictionary[cell_name][spike_name][round(10 ** num)][what_to_plot], "o")
        plt.show()

    @staticmethod
    def plotter_params(cell_name, spike_name, thresholds, dictionary, linear_regression: bool, log:bool):
        if linear_regression:
            param = ['r_square', 'fp']
        else:
            param = ["p", "r_2"]
        fig, ax = plt.subplots(nrows=1, ncols=len(param), figsize=(50, 6))
        if log:
            threshold = [round(10 ** i) for i in thresholds]
        else:
            threshold = [i for i in thresholds]
        for i, key in enumerate(param):
            perm_list = []
            for item in threshold:
                perm_list.append(dictionary[cell_name][spike_name][item][key])
            ax[i].plot(threshold, perm_list)
            ax[i].scatter(threshold, perm_list)
            ax[i].set_title(key)

        f_name = cell_name + "_" + spike_name + ".png"
        file_path = "../generated/" + f_name
        fig.savefig(file_path)


