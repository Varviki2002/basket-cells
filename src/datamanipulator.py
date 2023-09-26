import math
import pandas as pd


class DataManipulator:
    """
    A class to manage data manipulations.
    """

    def __init__(self, data):
        """
        :param data: The original dataframe.
        The dictionary look like this:
        {"the name of the cells":{"the number of the spike":{"relative firing-time:[], "IF":[]}}}
        """
        self.data = data
        self.dict = {}
        self.all_in_one_dict = {}
        self.__create_dict()
        self.names = [x[0] for x in self.data.columns][::2]

    def all_in_one_dict_creating(self, gbz_dict) -> dict:
        spike = 1
        for name in [x[0] for x in self.data.columns][::2]:
            for idx in range(0, len(self.data[name]["IF"])):
                if math.isnan(self.data[name]["IF"][idx]) is True:
                    spike = 1
                else:
                    writing = f"{spike}.spike"
                    if writing not in gbz_dict:
                        gbz_dict[writing] = {"relative firing time": [], "IF": []}
                    gbz_dict[writing]["relative firing time"].append(self.data[name]["relative firing time"][idx])
                    gbz_dict[writing]["IF"].append(self.data[name]["IF"][idx])
                    spike += 1
        return gbz_dict

    def create_cell_dict(self, gbz_dict) -> dict:
        spike = 1
        for name in [x[0] for x in self.data.columns][::2]:
            gbz_dict[name] = {}
            for idx in range(0, len(self.data[name]["IF"])):
                if math.isnan(self.data[name]["IF"][idx]) is True:
                    spike = 1
                else:
                    writing = f"{spike}.spike"
                    if writing not in gbz_dict[name]:
                        gbz_dict[name][writing] = {"relative firing time": [], "IF": []}
                    gbz_dict[name][writing]["relative firing time"].append(
                        self.data[name]["relative firing time"][idx])
                    gbz_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                    spike += 1
        return gbz_dict

    def __create_dict(self) -> None:
        """
        This method creates a dictionary from the data.
        :return None
        """
        self.all_in_one_dict_creating(gbz_dict=self.all_in_one_dict)
        self.create_cell_dict(gbz_dict=self.dict)

    def create_frame(self, cell_name: str, spike: str, y: bool, do_all: bool) -> pd.DataFrame:
        """
        This method creates a Dataframe from a cell's spike's relative firing time and IF.
        :param str cell_name: the name of the brain cell
        :param str spike: the number of the spike, for example "1.spike"
        :param bool y: if true the axes will be reversed
        :param bool do_all: if true the common dict from all cells will be used.
        :return pd.DataFrame: the created dataframe
        """
        if do_all:
            if not y:
                df = pd.DataFrame.from_dict(self.all_in_one_dict[spike])
                df.sort_values(by="relative firing time", ascending=False, inplace=True)
                return df
            else:
                df = pd.DataFrame.from_dict(self.all_in_one_dict[spike])
                df.sort_values(by="IF", ascending=False, inplace=True)
                return df
        else:
            if not y:
                df = pd.DataFrame.from_dict(self.dict[cell_name][spike])
                df.sort_values(by="relative firing time", ascending=False, inplace=True)
                return df
            else:
                df = pd.DataFrame.from_dict(self.dict[cell_name][spike])
                df.sort_values(by="IF", ascending=False, inplace=True)
                return df

    def measurements(self) -> pd.DataFrame:
        """
        Creates the measurement dataframe, that counts the number of the measurements in every spike of a cell.
        :return pd.DataFrame: the dataframe
        """
        measure_dict = {}
        for name in [x[0] for x in self.data.columns][::2]:
            measure_dict[name] = {}
            measure = 1
            for idx in range(0, len(self.data[name]["relative firing time"])):
                if idx == len(self.data[name]["relative firing time"]) - 1:
                    continue
                else:
                    if math.isnan(self.data[name]["relative firing time"][idx + 1]) is True:
                        continue
                    elif self.data[name]["relative firing time"][idx] < self.data[name]["relative firing time"][idx + 1]:
                        writing = f"{measure}.measure"
                        if writing not in measure_dict[name]:
                            measure_dict[name][writing] = {"relative firing time": [], "IF": []}
                        if math.isnan(self.data[name]["IF"][idx]) is False:
                            measure_dict[name][writing] = {"relative firing time": [], "IF": []}
                            measure_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                            measure_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                        else:
                            measure_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                    else:
                        writing = f"{measure}.measure"
                        if writing not in measure_dict[name]:
                            measure_dict[name][writing] = {"relative firing time": [], "IF": []}
                        if math.isnan(self.data[name]["IF"][idx]) is False:
                            measure_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                            measure_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                        else:
                            measure_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                        measure += 1

        n_dict = {}
        for name in [x[0] for x in self.data.columns][::2]:
            n_dict[name] = {}
            for spike in range(1, len(measure_dict[name]) + 1):
                writing = f"{spike}.measure"
                n_dict[name][writing] = len(measure_dict[name][writing]["relative firing time"])

        return pd.DataFrame(n_dict).transpose().fillna(0)

    def create_spike_frame(self) -> pd.DataFrame:
        """
        Creates a dataframe from the cell dictionary.
        :return pd.DataFrame: the created dataframe
        """
        n_dict = {}
        for name in [x[0] for x in self.data.columns][::2]:
            n_dict[name] = {}
            for spike in range(1, len(self.dict[name]) + 1):
                writing = f"{spike}.spike"
                n_dict[name][writing] = len(self.dict[name][writing]["relative firing time"])
        return pd.DataFrame(n_dict).transpose().fillna(0)
