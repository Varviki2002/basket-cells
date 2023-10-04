from lmfit import Minimizer, Parameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


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

    def create_lmfit_curve_fit(self, cell_name: str, plot_name: str, func_class,
                               do_all: bool, show: bool, switch_axes: bool, name_to_save: str,
                               param_values: tuple, log: bool) -> pd.DataFrame:
        """
        This method makes the curve fitting to the points of the given spike's cells.
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
        df_n = pd.DataFrame(index=letters[:func_class.n_params], columns=["1", "2", "3", "4", "5"]).fillna(0)
        if name_to_save not in self.func_dict:
            self.func_dict[name_to_save] = dict()
        self.func_dict[name_to_save][cell_name] = dict()
        for spike in range(5):
            string = f"{spike + 1}.spike"
            x, data = self.data_class.define_axes(cell_name=cell_name, string=string,
                                                  do_all=do_all, log=log, switch_axes=switch_axes)

            result = self.fit_the_function(func_class=func_class, param_values=param_values, x=x, data=data)
            print(type(result))
            if cell_name not in self.coeff:
                self.coeff[cell_name] = dict()
            self.coeff[cell_name][string] = dict()
            self.coeff[cell_name][string][name_to_save] = dict()
            self.coeff[cell_name][string][name_to_save]["params"] = list(result.params.valuesdict().values())
            r_2 = 1 - result.residual.var() / np.var(data)
            self.coeff[cell_name][string][name_to_save]["r_2"] = r_2
            self.coeff[cell_name][string][name_to_save]["uvars"] = result.uvars
            self.coeff[cell_name][string][name_to_save]["aic"] = result.aic
            self.coeff[cell_name][string][name_to_save]["bic"] = result.bic
            final = func_class(params=result.params, x=np.linspace(np.min(x), np.max(x), 201))

            self.coeff[cell_name][string][name_to_save]["r_2_sklearn"] = r2_score(y_true=data,
                                                                                  y_pred=func_class(params=result.params,
                                                                                                    x=x))

            # self.func_dict[name_to_save][cell_name][string] = None
            self.func_dict[name_to_save][cell_name][string] = func_class(params=result.params, x=x)

            if show:
                # report_fit(result)
                df_n = self.show_the_fit_results(df=df_n, num_params=func_class.n_params, result=result, spike=spike)
                # plot results
                if log:
                    self.plotter.plot_fitted_data(x=x, data=data, final=final, log=log,
                                                  spike=spike, plot_name=plot_name)
                else:
                    self.plotter.plot_fitted_data(x=x, data=data, final=final, log=log,
                                                  spike=spike, plot_name=plot_name)
            else:
                pass
        if show:
            return df_n

    def fit_the_function(self, func_class, param_values, x, data):

        def func_min(ps, x_data, dat):
            model = func_class(params=ps, x=x_data)
            return model - dat

        params = self.create_parameters(num_params=func_class.n_params, fit_params=param_values)

        # do fit, here with least_squares model
        minner = Minimizer(func_min, params, fcn_args=(x, data))
        result = minner.minimize(method="least_squares")

        return result

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
            df.loc[letters[i], str(spike + 1)] = result.params.valuesdict()[string]
        return df
