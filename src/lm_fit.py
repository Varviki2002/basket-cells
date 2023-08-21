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
        self.names = data_class.names
        self.letter = ["a", "b", "c", "d"]
        self.func_1 = dict()
        self.func_2 = dict()
        self.func_3 = dict()
        self.func_4 = dict()
        self.func_5 = dict()
        self.func_6 = dict()
        self.func_7 = dict()
        self.func_8 = dict()

    def create_lmfit_curve_fit(self, num_params: int, cell_name: str, name: str, function,
                               all: bool, func: int, show: bool, y: bool) -> pd.DataFrame:
        """
        This method makes the curve fitting to the points of the given spike's cells.
        :param int num_params: the number of the parameters
        :param str cell_name: the name of the cell
        :param str name: the name of the plot
        :param function: the function that includes the equation of the curve to be fitted
        :param bool all: if true the common dictionary of the cells will be used
        :param int func: the number of the dictionary, where the fitted data will be saved
        :param bool show: if true the curve fitting will be plotted
        :param bool y: if true the axes will be inverted
        :return -> pd.DataFrames: the parameters and their values will be shown
        """
        df_n = pd.DataFrame(index=self.letter[:num_params], columns=["1", "2", "3", "4", "5"]).fillna(0)
        colors = ["r", "b", "g", "mediumpurple", "gold"]
        function_colors = ["maroon", "midnightblue", "darkgreen", "darkmagenta", "darkorange"]
        for spike in range(5):
            string = f"{spike + 1}.spike"
            if all:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, all=all)
            else:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, all=all)

            if y:
                data = np.array(df["relative firing time"])
                x = np.array(df["IF"])
            else:
                x = np.array(df["relative firing time"])
                data = np.array(df["IF"])

            def func_min(params, x, data):
                model = function(params=params, x=x)
                return model - data

            # create a set of Parameters
            params = Parameters()
            if num_params == 1:
                params.add("a_param", value=2, min=0)
            elif num_params == 2:
                params.add("a_param", value=2, min=0)
                params.add("b_param", value=1, min=0)
            elif num_params == 3:
                params.add("a_param", value=2, min=0)
                params.add("b_param", value=1, min=0)
                params.add("c_param", value=1, min=0)
            else:
                params.add("a_param", value=2, min=0)
                params.add("b_param", value=1, min=0)
                params.add("c_param", value=1, min=0)
                params.add("d_param", value=1, min=0)

            # do fit, here with least_squares model
            minner = Minimizer(func_min, params, fcn_args=(x, data))
            result = minner.minimize(method="least_squares")

            # calculate final result
            # final = data + result.residual
            final = function(params=result.params,
                             x=np.linspace(np.min(x), np.max(x), 201))

            if y:
                if func == 5:
                    if spike == 0:
                        self.func_5[cell_name] = {}
                    if string in self.func_5[cell_name]:
                        self.func_5[cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_5[cell_name][string] = None
                        self.func_5[cell_name][string] = function(params=result.params, x=x)
                elif func == 6:
                    if spike == 0:
                        self.func_6[cell_name] = {}
                    if string in self.func_6[cell_name]:
                        self.func_6[cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_6[cell_name][string] = None
                        self.func_6[cell_name][string] = function(params=result.params, x=x)
                elif func is None:
                    pass
            else:
                if func == 1:
                    if spike == 0:
                        self.func_1[cell_name] = {}
                    if string in self.func_1[cell_name]:
                        self.func_1[cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_1[cell_name][string] = None
                        self.func_1[cell_name][string] = function(params=result.params, x=x)
                elif func == 2:
                    if spike == 0:
                        self.func_2[cell_name] = {}
                    if string in self.func_2[cell_name]:
                        self.func_2[cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_2[cell_name][string] = None
                        self.func_2[cell_name][string] = function(params=result.params, x=x)
                elif func is None:
                    pass

            if show:
                # report_fit(result)
                if num_params == 1:
                    df_n.loc["a", str(spike + 1)] = result.params.valuesdict()["a_param"]
                elif num_params == 2:
                    df_n.loc["a", str(spike + 1)] = result.params.valuesdict()["a_param"]
                    df_n.loc["b", str(spike + 1)] = result.params.valuesdict()["b_param"]
                elif num_params == 3:
                    df_n.loc["a", str(spike + 1)] = result.params.valuesdict()["a_param"]
                    df_n.loc["b", str(spike + 1)] = result.params.valuesdict()["b_param"]
                    df_n.loc["c", str(spike + 1)] = result.params.valuesdict()["c_param"]
                else:
                    df_n.loc["a", str(spike + 1)] = result.params.valuesdict()["a_param"]
                    df_n.loc["b", str(spike + 1)] = result.params.valuesdict()["b_param"]
                    df_n.loc["c", str(spike + 1)] = result.params.valuesdict()["c_param"]
                    df_n.loc["d", str(spike + 1)] = result.params.valuesdict()["d_param"]

                # plot results
                plt.plot(x, data, 'o', c=colors[spike])
                plt.plot(np.linspace(np.min(x), np.max(x), 201), final, 'r', c=function_colors[spike])
                plt.title(name)
                # plt.show()

            else:
                pass
        if show:
            return df_n

    def curve_fit_linear(self, num_params: int, cell_name: str, name: str, function,
                         all: bool, func: int, show: bool, y: bool) -> pd.DataFrame:
        """
        This method makes the curve fitting to the points of the given spike's cells
        in double logarithmic coordinate system.
        :param int num_params: the number of the parameters
        :param str cell_name: the name of the cell
        :param str name: the name of the plot
        :param function: the function that includes the equation of the curve to be fitted
        :param bool all: if true the common dictionary of the cells will be used
        :param int func: the number of the dictionary, where the fitted data will be saved
        :param bool show: if true the curve fitting will be plotted
        :param bool y: if true the axes will be inverted
        :return -> pd.DataFrames: the parameters and their values will be shown
        """
        df_n = pd.DataFrame(index=self.letter[:num_params], columns=["1", "2", "3", "4", "5"]).fillna(0)
        colors = ["r", "b", "g", "mediumpurple", "gold"]
        function_colors = ["maroon", "midnightblue", "darkgreen", "darkmagenta", "darkorange"]
        for spike in range(5):
            string = f"{spike + 1}.spike"
            if all:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, all=all)
            else:
                df = self.data_class.create_frame(cell_name=cell_name, spike=string, y=False, all=all)

            if y:
                data = np.log10(df["relative firing time"])
                x = np.log10(df["IF"])
            else:
                x = np.log10(df["relative firing time"])
                data = np.log10(df["IF"])

            def func_min(params, x, data):
                model = function(params=params, x=x)
                return model - data

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
                        self.func_7[cell_name] = {}
                    if string in self.func_7[cell_name]:
                        self.func_7[cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_7[cell_name][string] = None
                        self.func_7[cell_name][string] = function(params=result.params, x=x)
                elif func == 8:
                    if spike == 0:
                        self.func_8[cell_name] = {}
                    if string in self.func_8[cell_name]:
                        self.func_8[cell_name][string] = function(params=result.params, x=x)
                    else:
                        self.func_8[cell_name][string] = None
                        self.func_8[cell_name][string] = function(params=result.params, x=x)
                elif func is None:
                    pass
            else:
                if func == 4:
                    if spike == 0:
                        self.func_4[cell_name] = {}
                    if string in self.func_4[cell_name]:
                        self.func_4[cell_name][string] = function(params=result.params, x=x)
                        self.func_4[cell_name][string] = 10 ** self.func_4[cell_name][string].values
                    else:
                        self.func_4[cell_name][string] = None
                        self.func_4[cell_name][string] = function(params=result.params, x=x)
                        self.func_4[cell_name][string] = 10 ** self.func_4[cell_name][string].values
                elif func == 3:
                    if spike == 0:
                        self.func_3[cell_name] = {}
                    if string in self.func_3[cell_name]:
                        self.func_3[cell_name][string] = function(params=result.params, x=x)
                        self.func_3[cell_name][string] = 10 ** self.func_3[cell_name][string].values
                    else:
                        self.func_3[cell_name][string] = None
                        self.func_3[cell_name][string] = function(params=result.params, x=x)
                        self.func_3[cell_name][string] = 10 ** self.func_3[cell_name][string].values
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
            return df_n


