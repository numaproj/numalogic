import pandas as pd
from pynumaflow.mapper import Messages, Message, Datum, Mapper

from numalogic.synthetic import SyntheticTSGenerator


def getConfig():
    # ToDo: Load configs from a file
    return {
        "uniqueKeys": 10,
        "windowSize": 12,
        "testSize": 50,
        "configTags": [1, 2],
        "SyntheticTSGeneratorConfigs": [
            {
                "seq_len": 501,
                "num_series": 1,
                "freq": "T",
                "primary_period": 720,
                "secondary_period": 600,
                "seasonal_ts_prob": 1.0,
                "baseline_range": [500.0, 550.0],
                "slope_range": [-0.001, 0.01],
                "amplitude_range": [100, 200],
                "cosine_ratio_range": [0.5, 0.9],
                "noise_range": [5, 15],
            },
            {
                "seq_len": 501,
                "num_series": 1,
                "freq": "T",
                "primary_period": 720,
                "secondary_period": 600,
                "seasonal_ts_prob": 1.0,
                "baseline_range": [0.0, 1.0],
                "slope_range": [0.00000001, 0.0000001],
                "amplitude_range": [0.0011, 0.0033],
                "cosine_ratio_range": [0.5, 0.9],
                "noise_range": [0.0000001, 0.0000002],
            }
        ]
    }




def getSyntheticDataFromConfig(synthetic_ts_conf, test_size=5):
    ts_generator = SyntheticTSGenerator(**synthetic_ts_conf)
    ts_df = ts_generator.gen_tseries()

    # convert the index to unix timestamp
    ts_df.index = ts_df.index.astype("int64") // 10 ** 9
    # change the column name from s1 to failed
    ts_df.rename(columns={"s1": "failed"}, inplace=True)
    return ts_df


class LoadTestDataConnector:
    def __init__(self):
        self.config = getConfig()
        self.tot_ts = len(self.config["SyntheticTSGeneratorConfigs"])
        self.unique_keys = self.config["uniqueKeys"]
        self.mp = {}
        for k in range(1, self.tot_ts + 1):
            self.mp[k] = {"keys": [], "data": []}

        for i in range(1, self.unique_keys + 1):
            k = (i % self.tot_ts) + 1
            self.mp[k]["keys"].append(i)

        for k, synthetic_ts_conf in enumerate(self.config["SyntheticTSGeneratorConfigs"]):
            self.mp[k + 1]["data"] = getSyntheticDataFromConfig(synthetic_ts_conf)

        print(self.mp)

    def my_handler(self, id):
        for k in self.mp:
            if id in self.mp[k]["keys"]:
                print("shape of data used for training: ", self.mp[k]["data"].shape)
                return self.mp[k]["data"]


# if __name__ == "__main__":
#     obj = LoadTestDataConnector()
#     # print(obj.mp)
#     for i in range(0, 12):
#         obj.my_handler(i+1)
