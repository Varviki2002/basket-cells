import math
import pandas as pd


class DataManipulator:
    def __init__(self, data):
        self.data = data
        self.dict = self.create_dict(all_dict=False)

    def create_dict(self, all_dict: bool) -> dict:
        if all_dict:
            spike = 1
            gbz_dict = {}
            for name in [x[0] for x in self.data.columns][::2]:
                gbz_dict[name] = {}
                for idx in range(0, len(self.data[name]["IF"])):
                    if math.isnan(self.data[name]["IF"][idx]) is True:
                        spike = 1
                    else:
                        writing = f"{spike}.spike"
                        if writing in gbz_dict[name]:
                            gbz_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                            gbz_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                        else:
                            gbz_dict[name][writing] = {"relative firing time": [], "IF": []}
                            gbz_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                            gbz_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                        spike += 1
            return gbz_dict
        else:
            spike = 1
            gbz_dict = {}
            for name in [x[0] for x in self.data.columns][::2]:
                gbz_dict[name] = {}
                for idx in range(0, len(self.data[name]["IF"])):
                    if math.isnan(self.data[name]["IF"][idx]) is True:
                        spike = 1
                    else:
                        writing = f"{spike}.spike"
                        if writing in gbz_dict[name]:
                            gbz_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                            gbz_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                        else:
                            gbz_dict[name][writing] = {"relative firing time": [], "IF": []}
                            gbz_dict[name][writing]["relative firing time"].append(
                                self.data[name]["relative firing time"][idx])
                            gbz_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                        spike += 1
            return gbz_dict

    def create_frame(self, new_dict: dict, spike: str, y: bool):
        if not y:
            df = pd.DataFrame.from_dict(new_dict[spike])
            df.sort_values(by="relative firing time", ascending=False, inplace=True)
            return df
        else:
            df = pd.DataFrame.from_dict(new_dict[spike])
            df.sort_values(by="relative firing time", ascending=False, inplace=True)
            return df

    def measurements(self):
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
                        if writing in measure_dict[name]:
                            if math.isnan(self.data[name]["IF"][idx]) is False:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                                measure_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                            else:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                        else:
                            measure_dict[name][writing] = {"relative firing time": [], "IF": []}
                            if math.isnan(self.data[name]["IF"][idx]) is False:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                                measure_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                            else:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                    else:
                        writing = f"{measure}.measure"
                        if writing in measure_dict[name]:
                            if math.isnan(self.data[name]["IF"][idx]) is False:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                                measure_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                            else:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                        else:
                            measure_dict[name][writing] = {"relative firing time": [], "IF": []}
                            if math.isnan(self.data[name]["IF"][idx]) is False:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                                measure_dict[name][writing]["IF"].append(self.data[name]["IF"][idx])
                            else:
                                measure_dict[name][writing]["relative firing time"].append(
                                    self.data[name]["relative firing time"][idx])
                        measure += 1
        return measure_dict


