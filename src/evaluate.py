import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.datamanipulator import DataManipulator


class Evaluate:
    def __init__(self, data_class: DataManipulator, data_dict, func_1, func_2, func_3, func_4):
        self.data_class = data_class
        self.data_dict = data_dict
        self.func_1 = func_1
        self.func_2 = func_2
        self.func_3 = func_3
        self.func_4 = func_4

    def absolute_difference(self, cell_name: str, spike: str) -> list:
        string = f"{spike}.spike"
        abs_1 = np.abs(self.data_class.create_frame(cell_name=cell_name,
                                                    spike=string,
                                                    y=False,
                                                    all=False)["IF"] - self.func_1[cell_name][string])
        abs_2 = np.abs(self.data_class.create_frame(cell_name=cell_name,
                                                    spike=string,
                                                    y=False,
                                                    all=False)["IF"] - self.func_2[cell_name][string])
        abs_3 = np.abs(self.data_class.create_frame(cell_name=cell_name,
                                                    spike=string,
                                                    y=False,
                                                    all=False)["IF"] - self.func_3[cell_name][string])
        abs_4 = np.abs(self.data_class.create_frame(cell_name=cell_name,
                                                    spike=string,
                                                    y=False,
                                                    all=False)["IF"] - self.func_4[cell_name][string])

        print(f"The 1st fit difference: {abs_1}")
        print(f"The 2nd fit difference: {abs_2}")
        print(f"The 3rd fit difference: {abs_3}")
        print(f"The 4th fit difference: {abs_4}")

        plt.plot(abs_1, 'o', c='r')
        plt.plot(abs_2, 'o', c='g')
        plt.plot(abs_3, 'o', c='b')
        plt.plot(abs_4, 'o', c='darkmagenta')
        plt.legend()
        plt.title("Absolute difference for each point")

        print(f"The 1st fit sum: {np.sum(abs_1)}")
        print(f"The 1st fit sum: {np.sum(abs_2)}")
        print(f"The 1st fit sum: {np.sum(abs_3)}")
        print(f"The 1st fit sum: {np.sum(abs_4)}")

        smallest = []

        for i in range(len(abs_1)):
            minimum = np.min(np.array([abs_1[i], abs_2[i], abs_3[i], abs_4[i]]))
            if minimum == abs_1[i]:
                smallest.append("func_1")
            elif minimum == abs_2[i]:
                smallest.append("func_2")
            elif minimum == abs_3[i]:
                smallest.append("func_3")
            else:
                smallest.append("func_4")

        return smallest
