import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
import torch

def find_outliers(col, col_name, df_name, threshold=3, period=None, plot=True):
    stl = STL(col, period=period)
    result = stl.fit()
    residual = result.resid
    z_scores = (residual - residual.mean()) / residual.std()

    # detect outliers 3 stds from the mean
    outliers = col[abs(z_scores) > threshold]
    return outliers


def interpolate_outliers(time_series, outlier_indexes):
    for idx in outlier_indexes:
        interpolated_value = np.average(
            [
                time_series.loc[idx - 2 * timedelta(1)],
                time_series.loc[idx - 1 * timedelta(1)],
                time_series.loc[idx + 1 * timedelta(1)],
                time_series.loc[idx + 2 * timedelta(1)],
            ]
        )
        time_series.loc[idx] = interpolated_value


def load_cleaned_data(remove_outliers=True, validation_split=0.2):
    # loading data
    df_test = pd.read_csv(
        "data/DailyDelhiClimateTest.csv", index_col="date", parse_dates=True
    )
    df_train = pd.read_csv(
        "data/DailyDelhiClimateTrain.csv", index_col="date", parse_dates=True
    )


    # handling outliers in meanpressure
    for index in df_train[df_train["meanpressure"] > 1070].index:
        prev_value = df_train.loc[index - 1 * timedelta(1), "meanpressure"]
        next_value = df_train.loc[index + 1 * timedelta(1), "meanpressure"]
        average = (prev_value + next_value) / 2
        df_train.loc[index, "meanpressure"] = average

    for index in df_train[df_train["meanpressure"] < 980].index:
        prev_value = df_train.loc[index - 1 * timedelta(1), "meanpressure"]
        next_value = df_train.loc[index + 1 * timedelta(1), "meanpressure"]
        average = (prev_value + next_value) / 2
        df_train.loc[index, "meanpressure"] = average

    # first row is an outlier and no great way of interpolating it, so I just copy next value into it
    df_test["meanpressure"].iloc[0] = df_test["meanpressure"].iloc[1]

    # handling outliers in overall
    if remove_outliers:
        cols = [
            ("meantemp", "Mean Temperature"),
            ("humidity", "Humidity"),
            ("meanpressure", "Mean Pressure"),
            ("wind_speed", "Wind Speed"),
        ]
        datasets = [("train", df_train), ("test", df_test)]
        for col_idx, col_name in cols:
            for df_name, df in datasets:
                outliers = find_outliers(df[col_idx], col_name, df_name)
                interpolate_outliers(df[col_idx], outliers.index)

    num_validation_rows = int(validation_split * len(df_train))
    df_validation = df_train.head(num_validation_rows)
    df_train = df_train.iloc[num_validation_rows:]

    
    mu = df_train.mean()
    std = df_train.std()

    # mu = mu.drop("date")
    # std = std.drop("date")

    df_train = (df_train - mu) / std
    df_test = (df_test - mu) / std
    df_validation = (df_validation - mu) / std

    return df_train, df_test, df_validation, mu, std


def create_sequences(data, seq_length, target_length, start_token):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - target_length - 1):
        _x = data[i : i + seq_length]
        _y = []
        if start_token is not None:
            st =np.array(start_token).reshape(1, -1)
            _y = np.concatenate([st, data[i+seq_length:i+seq_length+target_length]])
        else:
            _y = data[i + seq_length : i + seq_length + target_length]
        
        xs.append(_x)
        ys.append(_y)
    return np.array(xs), np.array(ys)



class ClimateDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_length, target_length = 1, start_token = None) -> None:
        super().__init__()
        self.df = df
        self.sequences, self.targets = create_sequences(self.df.values, seq_length, target_length, start_token)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        x = self.sequences[index]
        y = self.targets[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

