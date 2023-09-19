import numpy as np
import matplotlib.pyplot as plt

from src.datamanipulator import DataManipulator
from src.lm_fit import LMFit


class Evaluate:
    """
    This class makes the evaluation.
    """

    def __init__(self, data_class: DataManipulator, lm_fit: LMFit) -> None:
        """
        :param data_class: the DataManipulator class
        :param lm_fit: the LMFit class
        """
        self.data_class = data_class
        self.lm_fit = lm_fit
        self.evaluate = dict()

    def absolute_difference_count(self, cell_name, string, string_name, func_name, y):
        self.evaluate[string_name] = np.abs(self.data_class.create_frame(cell_name=cell_name,
                                                                         spike=string, y=y,
                                                                         do_all=False)["relative firing time"] - \
                                            self.lm_fit.func_dict[func_name][cell_name][string])

    def absolute_difference(self, cell_name: str, spike: str, y: bool) -> list:
        string = f"{spike}.spike"
        if y:
            for i in range(4):
                string_name = f"abs_{i + 5}"
                func_name = f"func_{i + 5}"
                self.absolute_difference_count(cell_name=cell_name, string=string, string_name=string_name,
                                               func_name=func_name, y=y)
        else:
            for i in range(4):
                string_name = f"abs_{i + 1}"
                func_name = f"func_{i + 1}"
                self.absolute_difference_count(cell_name=cell_name, string=string, string_name=string_name,
                                               func_name=func_name, y=y)

        for i in range(4):
            string_name = f"abs_{i + 1}"
            print(f"The {i + 1}. fit difference: {self.evaluate[string_name]}")
        if not y:
            plt.plot(self.evaluate["abs_1"], 'o', c='r')
            plt.plot(self.evaluate["abs_2"], 'o', c='g')
            plt.plot(self.evaluate["abs_3"], 'o', c='b')
            plt.plot(self.evaluate["abs_4"], 'o', c='darkmagenta')
            plt.legend()
            plt.title("Absolute difference for each point")
            plt.show()
        else:
            plt.plot(self.evaluate["abs_5"], 'o', c='r')
            plt.plot(self.evaluate["abs_6"], 'o', c='g')
            plt.plot(self.evaluate["abs_7"], 'o', c='b')
            plt.plot(self.evaluate["abs_8"], 'o', c='darkmagenta')
            plt.legend()
            plt.title("Absolute difference for each point")
            plt.show()

        for i in range(4):
            string_name = f"abs_{i + 1}"
            print(f"The {i + 1}. fit sum: {np.sum(self.evaluate[string_name])}")

        smallest = []

        for i in range(len(self.evaluate["abs_1"])):
            minimum = np.min(np.array([self.evaluate["abs_1"][i], self.evaluate["abs_2"][i], self.evaluate["abs_3"][i],
                                       self.evaluate["abs_4"][i]]))
            if minimum == self.evaluate["abs_1"][i]:
                smallest.append("func_1")
            elif minimum == self.evaluate["abs_2"][i]:
                smallest.append("func_2")
            elif minimum == self.evaluate["abs_3"][i]:
                smallest.append("func_3")
            else:
                smallest.append("func_4")

        return smallest
