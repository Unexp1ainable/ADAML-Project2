import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler

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

    return df_train, df_test, df_validation
