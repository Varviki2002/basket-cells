from random import random
import pandas as pd
import matplotlib.pyplot as plt

from datamanipulator import DataManipulator


class Plotter(DataManipulator):
    def __init__(self, data):
        self.data = data
        super().__init__(data)

    def plot_spike(self, name: str, color: str) -> None:
        plt.scatter(self.dict[name]["relative firing time"], self.dict[name]["IF"], c=color)
        plt.title(name)
        plt.xlabel("relative firing time")
        plt.ylabel("IF")
        plt.show()

    def plot_dict_spikes(self):
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(self.dict))]
        for i in range(1, len(self.dict) + 1):
            spike = f"{i}.spike"
            self.plot_spike(name=spike, color=color[i - 1])

    def plot_frame_spike(self, name: str, spike_frame: pd.DataFrame, color: str) -> None:
        plt.scatter(spike_frame["relative firing time"], spike_frame["IF"],
                    c=color)
        plt.title(name)
        plt.xlabel("relative firing time")
        plt.ylabel("IF")
        plt.show()

