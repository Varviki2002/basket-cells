import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.datamanipulator import DataManipulator


class Evaluate:
    def __init__(self, data_dict, func_1, func_2, func_3, func_4):
        self.data_dict = data_dict
        self.func_1 = func_1
        self.func_2 = func_2
        self.func_3 = func_3
        self.func_4 = func_4

    def absolute_difference(self, cell_name: str, spike: str):
        string = f"{spike}.spike"
        abs_1 = np.abs(self.data_dict[cell_name][string]["IF"] - self.func_1[cell_name][string])
        abs_2 = np.abs(self.data_dict[cell_name][string]["IF"] - self.func_2[cell_name][string])
        abs_3 = np.abs(self.data_dict[cell_name][string]["IF"] - self.func_3[cell_name][string])
        abs_4 = np.abs(self.data_dict[cell_name][string]["IF"] - self.func_4[cell_name][string])

        print(f"The 1st fit difference: {abs_1}")
        print(f"The 2nd fit difference: {abs_2}")
        print(f"The 3rd fit difference: {abs_3}")
        print(f"The 4th fit difference: {abs_4}")

        plt.plot(abs_1, 'o', c='r')
        plt.plot(abs_2, 'o', c='g')
        plt.plot(abs_3, 'o', c='b')
        plt.plot(abs_4, 'o', c='darkmagenta')

        print(f"The 1st fit sum: {np.sum(abs_1)}")
        print(f"The 1st fit sum: {np.sum(abs_2)}")
        print(f"The 1st fit sum: {np.sum(abs_3)}")
        print(f"The 1st fit sum: {np.sum(abs_4)}")

