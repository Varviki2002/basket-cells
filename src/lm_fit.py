import json

from lmfit import Minimizer, Parameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

from src.datamanipulator import DataManipulator
from src.plot import Plotter


class LMFit:
    def __init__(self, data_class: DataManipulator, plotter: Plotter):
        self.plotter = plotter
        self.data_class = data_class
        self.data = data_class.data
        self.data_dict = data_class.dict
        self.all_in_one_dict = data_class.all_in_one_dict
        self.names = data_class.names
        self.func_dict = {}
        self.coeff = dict()
        self.bad_fitting = []
        self.df = pd.DataFrame()

    def create_lmfit_curve_fit(self, cell_name: str, plot_name: str, range_spike: int, func_class, param_values: tuple,
                               do_all: bool, log: bool, show: bool, switch_axes: bool,
                               name_to_save: str, save: bool, chosen_cells=None):
        """
        This method makes the curve fitting to the points of the given spike's cells.
        :param save:
        :param range_spike:
        :param chosen_cells:
        :param log:
        :param param_values:
        :param name_to_save:
        :param str cell_name: the name of the cell
        :param str plot_name: the name of the plot
        :param func_class: the func_class that includes the equation of the curve to be fitted
        :param bool do_all: if true the common dictionary of the cells will be used
        :param bool show: if true the curve fitting will be plotted
        :param bool switch_axes: if true the axes will be inverted
        :return -> pd.DataFrames: the parameters and their values will be shown
        """
        letters = ["a1", "a2", "a3", "a4"]
        df_n = pd.DataFrame(index=[i + 1 for i in range(range_spike)], columns=letters[:func_class.n_params]).fillna(0)
        df_params = pd.DataFrame(index=[i + 1 for i in range(range_spike)]).fillna(0)

        if name_to_save not in self.func_dict:
            self.func_dict[name_to_save] = dict()
        if not do_all and not isinstance(chosen_cells, list):
            if cell_name not in self.func_dict[name_to_save]:
                self.func_dict[name_to_save][cell_name] = dict()

        if cell_name not in self.coeff:
            self.coeff[name_to_save] = dict()

        for spike in range(0, range_spike):
            string = f"{spike + 1}.spike"
            x, data = self.data_class.define_axes(cell_name=cell_name, string=string,
                                                  do_all=do_all, chosen_cells=chosen_cells,
                                                  log=log, switch_axes=switch_axes)

            result, chi_sqr = self.fit_the_function(func_class=func_class, param_values=param_values, x=x, data=data)

            final = func_class(params=result.params, x=np.linspace(np.min(x), np.max(x), 201))

            chi2_stat = np.sum(result.residual ** 2 / func_class(params=result.params, x=x))
            # chi_square_test_statistic, p_value = stats.chisquare(data, func_class(params=result.params, x=x))

            if not do_all and not isinstance(chosen_cells, list):
                self.func_dict[name_to_save][cell_name][string] = func_class(params=result.params, x=x)
            else:
                self.func_dict[name_to_save][string] = func_class(params=result.params, x=x)

            if log:
                squared_difference = np.sum(
                    ((10 ** data) - (10 ** func_class(params=result.params, x=x))) ** 2 / len(data))
            else:
                squared_difference = np.sum((data - func_class(params=result.params, x=x)) ** 2 / len(data))
            if not do_all and not isinstance(chosen_cells, list):
                if cell_name not in self.coeff[name_to_save]:
                    self.coeff[name_to_save][cell_name] = dict()
                if string not in self.coeff[name_to_save][cell_name]:
                    self.coeff[name_to_save][cell_name][string] = dict()
                self.coeff[name_to_save][cell_name][string]["params"] = list(result.params.valuesdict().values())
                self.coeff[name_to_save][cell_name][string]["aic"] = result.aic
                self.coeff[name_to_save][cell_name][string]["bic"] = result.bic
                self.coeff[name_to_save][cell_name][string]["p-value"] = 1 - stats.chi2.cdf(chi2_stat, result.nfree)
                self.coeff[name_to_save][cell_name][string]["squared_diff"] = squared_difference
                self.coeff[name_to_save][cell_name][string]["r_2"] = r2_score(y_true=data, y_pred=func_class(
                    params=result.params,
                    x=x))
                self.coeff[name_to_save][cell_name][string]["adjusted_r_2"] = 1 - \
                                                                              (1 - r2_score(y_true=data,
                                                                                            y_pred=func_class(
                                                                                                params=result.params,
                                                                                                x=x))) * \
                                                                              (len(data) - 1) / (
                                                                                      len(data) - func_class.n_params - 1)
                self.coeff[name_to_save][cell_name][string]["RMSE"] = np.sqrt(mean_squared_error(y_true=data,
                                                                                                 y_pred=func_class(
                                                                                                     params=result.params,
                                                                                                     x=x)))

                if self.coeff[name_to_save][cell_name][string]["p-value"] < 0.05 and \
                        self.coeff[name_to_save][cell_name][string]["r_2"] < 0.6:
                    append_name = cell_name + str(spike)
                    self.bad_fitting.append(append_name)

            else:
                self.coeff[name_to_save][string] = dict()
                self.coeff[name_to_save][string]["params"] = list(result.params.valuesdict().values())
                self.coeff[name_to_save][string]["aic"] = result.aic
                self.coeff[name_to_save][string]["bic"] = result.bic
                self.coeff[name_to_save][string]["p-value"] = 1 - stats.chi2.cdf(chi2_stat, result.nfree)
                self.coeff[name_to_save][string]["squared_diff"] = squared_difference
                self.coeff[name_to_save][string]["r_2"] = r2_score(y_true=data, y_pred=func_class(
                    params=result.params,
                    x=x))
                self.coeff[name_to_save][string]["adjusted_r_2"] = 1 - \
                                                                   (1 - r2_score(y_true=data,
                                                                                 y_pred=func_class(params=result.params,
                                                                                                   x=x))) * \
                                                                   (len(data) - 1) / (
                                                                           len(data) - func_class.n_params - 1)
                self.coeff[name_to_save][string]["RMSE"] = np.sqrt(mean_squared_error(y_true=data, y_pred=func_class(
                    params=result.params, x=x)))

                if self.coeff[name_to_save][string]["p-value"] < 0.05 and self.coeff[name_to_save][string]["r_2"] < 0.6:
                    if do_all is True:
                        append_name = "all" + str(spike)
                    else:
                        append_name = str(chosen_cells) + str(spike)
                    self.bad_fitting.append(append_name)

            df_n = self.show_the_fit_results(df=df_n, num_params=func_class.n_params, result=result, spike=spike)
            df_params = self.show_the_param_results(df=df_params, num_params=func_class.n_params,
                                                    name_to_save=name_to_save, range_spike=range_spike,
                                                    do_all=do_all, cell_name=cell_name, spike=spike,
                                                    chosen_cells=chosen_cells)

            if show:
                # report_fit(result)
                # plot results
                if log:
                    self.plotter.plot_fitted_data(x=x, data=data, final=final, log=log,
                                                  spike=spike, plot_name=plot_name, range_spike=range_spike)

                else:
                    self.plotter.plot_fitted_data(x=x, data=data, final=final, log=log,
                                                  spike=spike, plot_name=plot_name, range_spike=range_spike)
            else:
                pass
        if show:
            plt.legend()
        if save:
            df_n_json = df_n.to_dict()
            df_params_json = df_params.to_dict()

            out = {'arguments of the function': {"cell_name":cell_name, "func_class": str(func_class),
                                                 'param_values': param_values, "chosen_cells": chosen_cells,
                                                 "do_all": do_all, "name_to_save": name_to_save},
                   'df_n': df_n_json,
                   'df_params': df_params_json,
                   }

            # evaluat = "evaluate" + plot_name + str(func_class.n_params) + '.xlsx'
            param_name = plot_name + str(func_class.n_params)
            file_param = "../generated/" + param_name
            # file_eval = "../generated/" + evaluat
            # df_n.to_excel(file_param)
            # df_params.to_excel(file_eval)
            out_file = open(f"{file_param}.json", "w")
            json.dump(out, out_file, indent=6)
            out_file.close()
            plt.savefig(file_param)
        if show:
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None,
                                   'display.precision', 3,
                                   ):
                print(df_params)
            return df_n

    def fit_the_function(self, func_class, param_values, x, data):

        def func_min(ps, x_data, dat):
            model = func_class(params=ps, x=x_data)
            return model - dat

        params = self.create_parameters(num_params=func_class.n_params, fit_params=param_values)

        # do fit, here with least_squares model
        minner = Minimizer(func_min, params, fcn_args=(x, data))
        result = minner.minimize(method="least_squares")
        chi_sqr = result.chisqr

        return result, chi_sqr

    @staticmethod
    def create_parameters(num_params: int, fit_params: tuple) -> Parameters:
        parameters = Parameters()
        for i in range(num_params):
            string = f"a{i + 1}_param"
            parameters.add(string, value=fit_params[0], min=fit_params[1])
        return parameters

    @staticmethod
    def show_the_fit_results(df: pd.DataFrame, num_params: int, result, spike: int) -> pd.DataFrame:
        letters = ["a1", "a2", "a3", "a4"]
        for i in range(num_params):
            string = f"a{i + 1}_param"
            df.loc[spike + 1, letters[i]] = result.params.valuesdict()[string]
        return df

    def show_the_param_results(self, df: pd.DataFrame, num_params: int, name_to_save, range_spike, do_all, cell_name,
                               spike, chosen_cells) -> pd.DataFrame:
        keys = ["p-value", "r_2", "adjusted_r_2", "RMSE", "aic", "bic", "squared_diff"]
        string = f"{spike + 1}.spike"
        for item in keys:
            if do_all:
                df.loc[spike + 1, item] = self.coeff[name_to_save][string][item]
            elif isinstance(chosen_cells, list):
                df.loc[spike + 1, item] = self.coeff[name_to_save][string][item]
            else:
                df.loc[spike + 1, item] = self.coeff[name_to_save][cell_name][string][item]
        return df
