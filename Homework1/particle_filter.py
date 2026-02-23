from kalman_filter import KalmanFilter
import pandas as pd
import numpy

DATA_FILE_PATH = "./data/Accelerometer.csv"

def get_approximate_dt():
    df_timestamps = pd.read_csv(DATA_FILE_PATH, header=0, usecols=["time"], converters={"time": str})
    series_diff = df_timestamps["time"].astype("int64").diff(periods=1)  # Break this into 2 steps because otherwise the Pandas default inferencing engine messes everything
    return series_diff.mean()

def get_approximate_noise():
    df_acceleration = pd.read_csv(DATA_FILE_PATH, header=0, usecols=["x", "y", "z"], converters={"x": float, "y": float, "z": float})
    variance = df_acceleration.var()
    return variance

noise_variance = get_approximate_noise()
print(noise_variance)