from kalman_filter import KalmanFilter
import pandas as pd
import numpy
import csv
from matplotlib import pyplot as plt

def get_trimmed_data(path: str, trim_factor: float = 0.05):
    data_df = pd.read_csv(path, header=0, converters={
        "time": str,    # Convert to string first otherwise Panda's default inferencing engine messes everything!
        "seconds_elapsed": float,
        "z": float,
        "y": float,
        "x": float
    })

    num_rows_to_trim = int(data_df.shape[0] * trim_factor)
    if 2 * num_rows_to_trim >= data_df.shape[0]:
        raise ValueError("trim_factor too high for data set")
    
    trimmed_df = data_df[num_rows_to_trim: -num_rows_to_trim].reset_index(drop=True)
    trimmed_df["time"] = trimmed_df["time"].astype("int64")
    trimmed_df[["z", "y", "x"]] = trimmed_df[["x", "y", "z"]]
    trimmed_df.rename(columns={
        "z": "x",
        "x": "z"
    }, inplace=True)
    return trimmed_df 

def get_average_dt_seconds(timestamps: pd.Series):
    NANOSECONDS_TO_SECONDS = 1e-9
    timestamps_seconds = timestamps * NANOSECONDS_TO_SECONDS
    return timestamps_seconds.diff(periods=1).mean()

def get_approximate_system_noise(stddev: float, dt: float):
    # https://cookierobotics.com/072/ seems to suggest that the entries in Q should be about how error scales with each component.
    # This means that acceleration should scale with itself, velocity should square with respect to acceleration, and position should be cubed with respect to acceleration, and so on...
    system_noise = numpy.zeros((9, 9), dtype=float)

    dt_squared = numpy.square(dt)
    dt_cubed = dt_squared * dt
    dt_fourth = numpy.square(dt_squared)
    system_noise_component = numpy.square(stddev) * numpy.array([
        [1.0, dt, dt_squared],
        [dt, dt_squared, dt_cubed],
        [dt_squared, dt_cubed, dt_fourth]
    ])

    for i in range(0, 9, 3):
        system_noise[i:i+3, i:i+3] = system_noise_component

    return system_noise

def main():
    DATA_FILE_PATH = "./still-data/Accelerometer.csv"
    df_data = get_trimmed_data(DATA_FILE_PATH)

    average_dt_seconds = get_average_dt_seconds(df_data["time"])
    system_noise = get_approximate_system_noise(0.001, average_dt_seconds)
    measurement_noise = df_data[["x", "y", "z"]].cov().to_numpy()
    kfilter = KalmanFilter(dt=average_dt_seconds, system_noise=system_noise, measurement_noise=measurement_noise)
    filtered_data = []
    # filtered_covariances = []

    for row in df_data.loc[:, ["x", "y", "z"]].itertuples(index=False):
        datum = numpy.array((row.x, row.y, row.z))
        kfilter.predict()
        kfilter.update(datum)

        filtered_data.append(kfilter.x.flatten())
    df_filtered = pd.DataFrame(filtered_data, columns=
        ["acc_x", "acc_y", "acc_z", "vel_x", "vel_y", "vel_z", "pos_x", "pos_y", "pos_z"]
    )

    for column_label, axis in zip(["vel_x"], ["x"]):
        plt.plot(df_data.loc[:, "seconds_elapsed"], df_data.loc[:, axis], marker="+", label=f"raw_{axis}")
        plt.plot(df_data.loc[:, "seconds_elapsed"], df_filtered.loc[:, column_label], marker=",", label=column_label)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
