import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from src.datamanipulator import DataManipulator
from src.lm_fit import LMFit
from src.plot import Plotter


class Evaluate:
    """
    This class makes the evaluation.
    """

    def __init__(self, data_class: DataManipulator, lm_fit: LMFit, plotter: Plotter) -> None:
        """
        :param data_class: the DataManipulator class
        :param lm_fit: the LMFit class
        """
        self.data_class = data_class
        self.lm_fit = lm_fit
        self.evaluate = dict()
        self.squared_diff_dict = dict()
        self.fit_parameters = dict()
        self.plotter = plotter
        self.bad_lin_regression = []
        self.multiindex_dictionary = {}

    def count_if_threshold(self, cell_name, spike_name, func_class, param_values, threshold, ax,
                           chosen_cells, linear_regression: bool, log: bool):
        if isinstance(chosen_cells, list):
            cell_name = "all"
        if cell_name not in self.fit_parameters:
            self.fit_parameters[cell_name] = dict()
        self.fit_parameters[cell_name][spike_name] = dict()
        if log or linear_regression:
            threshold = np.log10(threshold)
            dict_frame = np.log10(self.data_class.create_frame(cell_name=cell_name, spike=spike_name,
                                                               y=False, do_all=False, chosen_cells=chosen_cells))
        else:
            threshold = threshold
            dict_frame = self.data_class.create_frame(cell_name=cell_name, spike=spike_name,
                                                      y=False, do_all=False, chosen_cells=chosen_cells)

        for item, num in enumerate(threshold):
            if log or linear_regression:
                self.fit_parameters[cell_name][spike_name][round(10 ** num)] = dict()
            else:
                self.fit_parameters[cell_name][spike_name][num] = dict()
            df = dict_frame
            for i in range(len(dict_frame)-1, -1, -1):
                if dict_frame["IF"].iloc[i] > num:
                    df = df.drop(labels=i, axis=0)
            result, chisq = self.lm_fit.fit_the_function(func_class=func_class, param_values=param_values,
                                                         x=df["relative firing time"], data=df["IF"])
            final = func_class(params=result.params, x=df["relative firing time"])
            mean = np.mean(df["IF"])
            if linear_regression:
                r_2 = (np.sum((mean - df["IF"]) ** 2) - np.sum((df["IF"] - final) ** 2)) / np.sum(
                    (mean - df["IF"]) ** 2)
                p, r_square, conf_int, fp, f, params = self.linear_regression(x=df["relative firing time"], y=df["IF"])
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["p"] = p
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["r_square"] = r_square
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["conf_int"] = conf_int
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["r_2"] = r_2
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["fp"] = fp
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["f"] = f

            elif log is True and linear_regression is False:
                params = list(result.params.valuesdict().values())
                chi2_stat = np.sum(
                    result.residual ** 2 / func_class(params=result.params, x=df["relative firing time"]))
                # self.fit_parameters[cell_name][spike_name][round(10 ** num)]["params"] = list(result.params.valuesdict().values())
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["aic"] = result.aic
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["bic"] = result.bic
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["r_2"] = r2_score(y_true=df["IF"],
                                                                                               y_pred=func_class(
                                                                                                   params=result.params,
                                                                                                   x=df["relative firing time"]))
                self.fit_parameters[cell_name][spike_name][round(10 ** num)]["p"] = 1 - stats.chi2.cdf(chisq,
                                                                                                       result.nfree)

            else:
                params = list(result.params.valuesdict().values())
                chi2_stat = np.sum(result.residual ** 2 / func_class(params=result.params, x=df["relative firing time"]))
                # self.fit_parameters[cell_name][spike_name][num]["params"] = list(result.params.valuesdict().values())
                # self.fit_parameters[cell_name][spike_name][num]["chi_sqr"] = result.chisqr
                self.fit_parameters[cell_name][spike_name][num]["aic"] = result.aic
                self.fit_parameters[cell_name][spike_name][num]["bic"] = result.bic
                self.fit_parameters[cell_name][spike_name][num]["r_2"] = r2_score(y_true=df["IF"],
                                                                                  y_pred=func_class(
                                                                                      params=result.params,
                                                                                      x=df["relative firing time"]))
                self.fit_parameters[cell_name][spike_name][num]["p"] = 1 - stats.chi2.cdf(chisq,
                                                                                          result.nfree)

            if linear_regression:
                self.plotter.different_if_plotter(df=df, p=params, ax=ax, idx=item, threshold=threshold)
            elif log:
                self.plotter.plot_fitted_data_eval(x=df["relative firing time"], data=df["IF"], final=final, log=log,
                                                   plot_name=round(10 ** num), ax=ax, id=item)
            else:
                self.plotter.plot_fitted_data_eval(x=df["relative firing time"], data=df["IF"], final=final, log=log,
                                                   plot_name=num, ax=ax, id=item)
        if linear_regression:
            if self.fit_parameters[cell_name][spike_name][round(10 ** threshold[-1])]["fp"] > 0.05 and \
                    self.fit_parameters[cell_name][spike_name][round(10 ** threshold[-1])]["r_2"] < 0.6:
                append_name = "linear_regression" + cell_name + spike_name
                self.bad_lin_regression.append(append_name)
        else:
            if log:
                if self.fit_parameters[cell_name][spike_name][round(10 ** threshold[-1])]["p"] < 0.05 and \
                        self.fit_parameters[cell_name][spike_name][round(10 ** threshold[-1])]["r_2"] < 0.6:
                    append_name = "curve_fit" + "log" + cell_name + spike_name
                    self.bad_lin_regression.append(append_name)
            else:
                if self.fit_parameters[cell_name][spike_name][threshold[-1]]["p"] < 0.05 and \
                        self.fit_parameters[cell_name][spike_name][threshold[-1]]["r_2"] < 0.6:
                    append_name = "curve_fit" + cell_name + spike_name
                    self.bad_lin_regression.append(append_name)

        self.plotter.plotter_params(cell_name=cell_name, spike_name=spike_name, thresholds=threshold,
                                    dictionary=self.fit_parameters, linear_regression=linear_regression, log=log)

        if linear_regression:
            dictionary = {cell_name: {'fp': [self.fit_parameters[cell_name][spike_name][round(10 ** num)]["fp"] for num in threshold],
                                      'r_2': [self.fit_parameters[cell_name][spike_name][round(10 ** num)]["r_2"] for num in threshold]}}
        else:
            if log:
                dictionary = {
                    cell_name: {'p': [self.fit_parameters[cell_name][spike_name][round(10 ** num)]["p"] for num in threshold],
                                'r_2': [self.fit_parameters[cell_name][spike_name][round(10 ** num)]["r_2"] for num in threshold]}}
            else:
                dictionary = {
                    cell_name: {'p': [self.fit_parameters[cell_name][spike_name][num]["p"] for num in threshold],
                                'r_2': [self.fit_parameters[cell_name][spike_name][num]["r_2"] for num in threshold]}}
        reform = {(outerKey, innerKey): values for outerKey, innerDict in dictionary.items() for innerKey, values in
                  innerDict.items()}
        df = pd.DataFrame.from_dict(reform, orient='index').transpose()
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        evaluat = cell_name + str(func_class.n_params) + '.xlsx'
        file_eval = "../generated/" + evaluat
        df.to_excel(file_eval)

    @staticmethod
    def linear_regression(x, y):
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        result = model.fit()
        params = result.params
        p = result.pvalues
        r_square = result.rsquared
        conf_int = result.conf_int(alpha=0.05, cols=None)
        fp = result.f_pvalue
        f = result.fvalue
        return p, r_square, conf_int, fp, f, params

    def compare_values_to_original(self):
        pass

    """def squared_difference(self, string_name, cell_name, string, y, func_name, func_num):
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
        """

    """def count_if(self, cell_name, do_all, func_class, param_values=(2, 0)):
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
        """

    """def absolute_difference_count(self, cell_name, string, string_name, func_name, y):
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
        """

