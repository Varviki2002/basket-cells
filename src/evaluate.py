import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.datamanipulator import DataManipulator


class Evaluate:
    def __init__(self, data_dict, func_1, func_2, func_3, func_4):
        self.dict = data_dict
        self.func_1 = func_1
        self.func_2 = func_2
        self.func_3 = func_3
        self.func_4 = func_4

    def absolute_difference(self, cell_name: str, spike: str):
        string = f"{spike}.spike"
        abs_1 = np.abs(np.array(self.dict[cell_name][string]) - np.array(self.func_1[cell_name][string]))
        abs_2 = np.abs(np.array(self.dict[cell_name][string]) - np.array(self.func_2[cell_name][string]))
        abs_3 = np.abs(np.array(self.dict[cell_name][string]) - np.array(self.func_3[cell_name][string]))
        abs_4 = np.abs(np.array(self.dict[cell_name][string]) - np.array(self.func_4[cell_name][string]))

        print(f"The 1st fit difference: {abs_1}")
        print(f"The 2nd fit difference: {abs_2}")
        print(f"The 3rd fit difference: {abs_3}")
        print(f"The 4th fit difference: {abs_4}")

        plt.plot(np.linspace(1, len(abs_1) + 1, 1), abs_1, 'r')
        plt.plot(np.linspace(1, len(abs_2) + 1, 1), abs_2, 'g')
        plt.plot(np.linspace(1, len(abs_3) + 1, 1), abs_3, 'b')
        plt.plot(np.linspace(1, len(abs_4) + 1, 1), abs_4, 'darkmagenta')

