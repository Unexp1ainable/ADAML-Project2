import pandas as pd
from datetime import timedelta


def load_cleaned_data():
    # loading data
    df_test = pd.read_csv(
        "data/DailyDelhiClimateTest.csv", index_col="date", parse_dates=True
    )
    df_train = pd.read_csv(
        "data/DailyDelhiClimateTrain.csv", index_col="date", parse_dates=True
    )

    # handling outliers
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

    return df_train, df_test
