import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
        self.squared_diff_dict = dict()
        self.linear_regression_parameters = dict()

    def count_if_threshold(self, cell_name, spike_name, func_class, param_values):
        if cell_name not in self.linear_regression_parameters:
            self.linear_regression_parameters[cell_name] = dict()
        self.linear_regression_parameters[cell_name][spike_name] = dict()
        threshold = [50, 70, 90, 110, 130, 150, 170, 190]
        threshold = np.log10(threshold)
        rft = np.log10(self.data_class.dict[cell_name][spike_name]["relative firing time"])
        inst_f = np.log10(self.data_class.dict[cell_name][spike_name]["IF"])

        for num in threshold:
            self.linear_regression_parameters[cell_name][spike_name][num] = dict()
            for i in range(len(inst_f)):
                if inst_f[i] > num:
                    inst_f.remove(inst_f[i])
                    rft.remove(rft[i])

            result = self.lm_fit.fit_the_function(func_class=func_class, param_values=param_values,
                                                  x=rft, data=inst_f)
            final = func_class(params=result.params, x=rft)
            mean = np.mean(inst_f)
            r_2 = (np.sum((mean-inst_f) ** 2) - np.sum((inst_f - final) ** 2)) / np.sum((mean-inst_f) ** 2)
            p, r_square, conf_int = self.linear_regression(x=rft, y=inst_f)
            self.linear_regression_parameters[cell_name][spike_name][num]["p"] = p
            self.linear_regression_parameters[cell_name][spike_name][num]["r_square"] = r_square
            self.linear_regression_parameters[cell_name][spike_name][num]["conf_int"] = conf_int
            self.linear_regression_parameters[cell_name][spike_name][num]["r_2"] = r_2

    @staticmethod
    def linear_regression(x, y):
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        result = model.fit()
        p = result.pvalues
        r_square = result.rsquared
        conf_int = result.conf_int(alpha=0.05, cols=None)
        return p, r_square, conf_int

    def squared_difference(self, string_name, cell_name, string, y, func_name, func_num):
        if y:
            original_data = np.log10(self.data_class.create_frame(cell_name=cell_name, spike=string, y=y,
                                                                  do_all=False)["relative firing time"])
            if func_num == 1 or func_num == 2:
                fitted_data = np.log10(self.lm_fit.func_dict[func_name][cell_name][string])
            else:
                fitted_data = self.lm_fit.func_dict[func_name][cell_name][string]
        else:
            original_data = np.log10(self.data_class.create_frame(cell_name=cell_name, spike=string, y=y,
                                                                  do_all=False)["IF"])
            if func_name == 1 or func_name == 2:
                fitted_data = np.log10(self.lm_fit.func_dict[func_name][cell_name][string])
            else:
                fitted_data = self.lm_fit.func_dict[func_name][cell_name][string]

        self.evaluate[string_name] = (original_data - fitted_data) ** 2

    def count_if(self, cell_name, do_all, func_class, param_values=(2, 0)):
        string = f"{1}.spike"
        func_dict = self.data_class.dict[cell_name][string]
        count_threshold = list(np.arange(5, len(func_dict["IF"]), 10))
        count_threshold.append(len(func_dict["IF"]))
        if cell_name not in self.squared_diff_dict:
            self.squared_diff_dict[cell_name] = dict()
        for num in count_threshold:
            df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, do_all=do_all)
            x = np.log10(df["relative firing time"][0:int(num)])
            data = np.log10(df["IF"][0:int(num)])
            result = self.lm_fit.fit_the_function(func_class=func_class, param_values=param_values, x=x, data=data)
            final = func_class(params=result.params, x=x)
            self.squared_diff_dict[cell_name][str(num)] = (data - final) ** 2

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