from lmfit import Minimizer, Parameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.datamanipulator import DataManipulator


class LMFit:
    def __init__(self, data_class: DataManipulator):
        self.data_class = data_class
        self.data = data_class.data
        self.data_dict = data_class.dict
        self.all_in_one_dict = data_class.all_in_one_dict
        self.names = data_class.names
        self.letter = ["a1", "a2", "a3", "a4"]
        self.func_dict = {}

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
        df_n = pd.DataFrame(index=self.letter[:func_class.n_params], columns=["1", "2", "3", "4", "5"]).fillna(0)
        colors = ["r", "b", "g", "mediumpurple", "gold"]
        function_colors = ["maroon", "midnightblue", "darkgreen", "darkmagenta", "darkorange"]
        if name_to_save not in self.func_dict:
            self.func_dict[name_to_save] = dict()
        self.func_dict[name_to_save][cell_name] = dict()
        for spike in range(5):
            string = f"{spike + 1}.spike"
            if do_all:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, do_all=do_all)
            else:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, do_all=do_all)

            if log:
                if switch_axes:
                    data = np.log10(df["relative firing time"])
                    x = np.log10(df["IF"])
                else:
                    x = np.log10(df["relative firing time"])
                    data = np.log10(df["IF"])
            else:
                if switch_axes:
                    data = np.array(df["relative firing time"])
                    x = np.array(df["IF"])
                else:
                    x = np.array(df["relative firing time"])
                    data = np.array(df["IF"])

            result = self.fit_the_function(func_class=func_class, param_values=param_values, x=x, data=data)
            final = func_class(params=result.params, x=np.linspace(np.min(x), np.max(x), 201))

            # self.func_dict[name_to_save][cell_name][string] = None
            self.func_dict[name_to_save][cell_name][string] = func_class(params=result.params, x=x)

            if show:
                # report_fit(result)
                df_n = self.show_the_fit_results(df=df_n, num_params=func_class.n_params, result=result, spike=spike)
                # plot results
                if log:
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.plot(10 ** x, 10 ** data, 'o', c=colors[spike])
                    plt.plot(10 ** np.linspace(np.min(x), np.max(x), 201),
                             10 ** final,
                             'r', c=function_colors[spike])
                    plt.title(plot_name)
                else:
                    plt.plot(x, data, 'o', c=colors[spike])
                    plt.plot(np.linspace(np.min(x), np.max(x), 201), final, 'r', c=function_colors[spike])
                    plt.title(plot_name)
                    # plt.show()

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

    def show_the_fit_results(self, df: pd.DataFrame, num_params: int, result, spike: int) -> pd.DataFrame:
        for i in range(num_params):
            string = f"a{i + 1}_param"
            df.loc[self.letter[i], str(spike + 1)] = result.params.valuesdict()[string]
        return df

    """def curve_fit_linear(self, num_params: int, cell_name: str, name: str, function,
                         all: bool, func: int, show: bool, y: bool) -> pd.DataFrame:
        
        This method makes the curve fitting to the points of the given spike's cells
        in double logarithmic coordinate system.
        :param int num_params: the number of the parameters
        :param str cell_name: the name of the cell
        :param str name: the name of the plot
        :param function: the func_class that includes the equation of the curve to be fitted
        :param bool all: if true the common dictionary of the cells will be used
        :param int func: the number of the dictionary, where the fitted data will be saved
        :param bool show: if true the curve fitting will be plotted
        :param bool y: if true the axes will be inverted
        :return -> pd.DataFrames: the parameters and their values will be shown
        
        df_n = pd.DataFrame(index=self.letter[:num_params], columns=["1", "2", "3", "4", "5"]).fillna(0)
        colors = ["r", "b", "g", "mediumpurple", "gold"]
        function_colors = ["maroon", "midnightblue", "darkgreen", "darkmagenta", "darkorange"]
        for spike in range(5):
            string = f"{spike + 1}.spike"
            if all:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, do_all=all)
            else:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, do_all=all)

            if y:
                data = np.log10(df["relative firing time"])
                x = np.log10(df["IF"])
            else:
                x = np.log10(df["relative firing time"])
                data = np.log10(df["IF"])

            def func_min(ps, x_data, dat):
                model = function(params=ps, x=x_data)
                return model - dat

            # create a set of Parameters
            params = Parameters()
            params.add("a_param", value=2)
            params.add("b_param", value=1)

            # do fit, here with least_squares model
            minner = Minimizer(func_min, params, fcn_args=(x, data))
            result = minner.minimize(method="least_squares")

            # calculate final result
            # final = data + result.residual
            final = function(params=result.params, x=np.linspace(np.min(x), np.max(x), 201))

            if y:
                if func == 7:
                    if spike == 0:
                        self.func_dict["func_7"][cell_name] = {}
                    if string in self.func_dict["func_7"][cell_name]:
                        self.func_dict["func_7"][cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_dict["func_7"][cell_name][string] = None
                        self.func_dict["func_7"][cell_name][string] = function(params=result.params, x=x)
                elif func == 8:
                    if spike == 0:
                        self.func_dict["func_8"][cell_name] = {}
                    if string in self.func_dict["func_8"][cell_name]:
                        self.func_dict["func_8"][cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_dict["func_8"][cell_name][string] = None
                        self.func_dict["func_8"][cell_name][string] = function(params=result.params, x=x)
                elif func is None:
                    pass
            else:
                if func == 4:
                    if spike == 0:
                        self.func_dict["func_4"][cell_name] = {}
                    if string in self.func_dict["func_4"][cell_name]:
                        self.func_dict["func_4"][cell_name][string] = function(params=result.params, x=x)
                        self.func_dict["func_4"][cell_name][string] = \
                            10 ** self.func_dict["func_4"][cell_name][string].values
                    else:
                        self.func_dict["func_4"][cell_name][string] = None
                        self.func_dict["func_4"][cell_name][string] = function(params=result.params, x=x)
                        self.func_dict["func_4"][cell_name][string] = \
                            10 ** self.func_dict["func_4"][cell_name][string].values
                elif func == 3:
                    if spike == 0:
                        self.func_dict["func_3"][cell_name] = {}
                    if string in self.func_dict["func_3"][cell_name]:
                        self.func_dict["func_3"][cell_name][string] = function(params=result.params, x=x)
                        self.func_dict["func_3"][cell_name][string] = \
                            10 ** self.func_dict["func_3"][cell_name][string].values
                    else:
                        self.func_dict["func_3"][cell_name][string] = None
                        self.func_dict["func_3"][cell_name][string] = function(params=result.params, x=x)
                        self.func_dict["func_3"][cell_name][string] = \
                            10 ** self.func_dict["func_3"][cell_name][string].values
                elif func is None:
                    pass

            if show:

                # report_fit(result)
                if num_params == 1:
                    df_n.loc["a", str(spike + 1)] = result.params.valuesdict()["a_param"]
                elif num_params == 2:
                    df_n.loc["a", str(spike + 1)] = result.params.valuesdict()["a_param"]
                    df_n.loc["b", str(spike + 1)] = result.params.valuesdict()["b_param"]

                # plot results
                plt.xscale('log')
                plt.yscale('log')
                plt.plot(10 ** x,
                         10 ** data,
                         'o', c=colors[spike])
                plt.plot(10 ** np.linspace(np.min(x), np.max(x), 201),
                         10 ** final,
                         'r', c=function_colors[spike])
                plt.title(name)
        if show:
            return df_n"""
