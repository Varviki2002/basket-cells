from random import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.datamanipulator import DataManipulator


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

    def plot_spike(self, name: str, spike_name: str, color: str, all: bool) -> None:
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
        else:
            plt.scatter(self.dict[name][spike_name]["relative firing time"],
                        self.dict[name][spike_name]["IF"], c=color)
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

    def plot_frame_spike(self, name: str, spike_frame: pd.DataFrame, color: str) -> None:
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

    @staticmethod
    def plot_fitted_data(x, data, final, log, spike, plot_name):
        colors = ["r", "b", "g", "mediumpurple", "gold"]
        function_colors = ["maroon", "midnightblue", "darkgreen", "darkmagenta", "darkorange"]
        if log:
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(10 ** x, 10 ** data, 'o', c=colors[spike])
            plt.plot(10 ** np.linspace(np.min(x), np.max(x), 201),
                     10 ** final,
                     'r', c=function_colors[spike])
            plt.title(plot_name)
        else:
            plt.plot(x, data, 'o', c=colors[spike])
            plt.plot(np.linspace(np.min(x), np.max(x), 201), final, 'r', c=function_colors[spike])
            plt.title(plot_name)
            # plt.show()

    @staticmethod
    def different_if_plotter(df, p, ax, num):
        ax[num].scatter(df["relative firing time"], df["IF"])
        ax[num].plot(df["relative firing time"], p[0] + p[1]*df["relative firing time"])

    @staticmethod
    def plot_errors(dictionary: dict, threshold: list, what_to_plot: str, cell_name, spike_name):
        arang = np.arange(1, len(threshold)+1, 1)
        for i, num in enumerate(threshold):
            plt.plot(arang[i], dictionary[cell_name][spike_name][round(10 ** num)][what_to_plot], "o")
        plt.show()


