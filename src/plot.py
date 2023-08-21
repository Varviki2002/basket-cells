from random import random
import pandas as pd
import matplotlib.pyplot as plt

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
            all_dict = self.data_class.create_dict(all_in_one=True)
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

