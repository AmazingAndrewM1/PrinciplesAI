from filterpy.kalman import KalmanFilter
import numpy
import kalman_filter
import pathlib
import pandas as pd
from matplotlib import pyplot as plt

def get_approximate_covariance(sigma_v: float, sigma_p: float):
    covariance = numpy.zeros(shape=(6, 6), dtype=float)
    covariance[0:3, 0:3] = numpy.square(sigma_v) * numpy.eye(3, 3)
    covariance[3:6, 3:6] = numpy.square(sigma_p) * numpy.eye(3, 3)

    return covariance

def get_approximate_system_noise(stddev: float, dt: float):
    # https://cookierobotics.com/072/ seems to suggest that the entries in system noise should be about how error scales with each component.
    # This means that acceleration should scale with itself, velocity should square with respect to acceleration, and position should be cubed with respect to acceleration, and so on...
    system_noise = numpy.zeros((6, 6), dtype=float)

    dt_squared = numpy.square(dt)
    dt_cubed = dt_squared * dt
    dt_fourth = numpy.square(dt_squared)
    system_noise_component = numpy.square(stddev) * numpy.array([
        [dt_squared, dt_cubed],
        [dt_cubed, dt_fourth]
    ])

    for i in range(0, 6, 2):
        system_noise[i:i+2, i:i+2] = system_noise_component

    return system_noise

def get_approximate_velocity_covariance():
    precomputed_path = pathlib.Path() / "precomputed" / "velocity-covariance.npy"
    if precomputed_path.is_file():
        return numpy.load(precomputed_path)

    STILL_DATA_FILE_PATH = pathlib.Path() / "data" / "still" / "Accelerometer.csv"
    vel_history = [numpy.zeros(shape=(3,), dtype=float)]
    prev_t = 0.0
    df_data = kalman_filter.get_trimmed_data(STILL_DATA_FILE_PATH)
    for row in df_data.itertuples(index=False):
        accel_datum = numpy.array((row.x, row.y, row.z), dtype=float)
        curr_vel = vel_history[-1] + accel_datum * (row.seconds_elapsed - prev_t)
        vel_history.append(curr_vel.copy())
        prev_t = row.seconds_elapsed
    result = numpy.cov(numpy.vstack(vel_history, dtype=float), rowvar=False, ddof=1)
    numpy.save(precomputed_path, result)
    return result

def main():
    DATA_FILE_PATH = pathlib.Path() / "data" / "still" / "Accelerometer.csv"
    df_data: pd.DataFrame = kalman_filter.get_trimmed_data(DATA_FILE_PATH)
    average_dt_seconds: float = kalman_filter.get_average_dt_seconds(df_data["time"])
    covariance = get_approximate_covariance(0.1, 0.3)
    system_noise: numpy.ndarray = get_approximate_system_noise(0.001, average_dt_seconds)
    measurement_uncertainty = get_approximate_velocity_covariance()

    kfilter = KalmanFilter(dim_x=6, dim_z=3, dim_u=3)
    kfilter.x = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)             # vel_x, vel_y, vel_z, pos_x, pos_y, pos_z
    kfilter.P = covariance                      # covariance matrix
    kfilter.Q = system_noise                    # system noise
    kfilter.R = measurement_uncertainty         # measurement uncertainty
    kfilter.H = numpy.array([                   # Measurement matrix
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ], dtype=float)
    kfilter.F = numpy.array([                   # state-transition matrix
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [average_dt_seconds, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, average_dt_seconds, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, average_dt_seconds, 0.0, 0.0, 1.0]
    ], dtype=float)
    kfilter.B = numpy.array([                   # Input control matrix
        [0.5 * numpy.square(average_dt_seconds), 0.0, 0.0],
        [0.0, 0.5 * numpy.square(average_dt_seconds), 0.0],
        [0.0, 0.0, 0.5 * numpy.square(average_dt_seconds)],
        [average_dt_seconds, 0.0, 0.0],
        [0.0, average_dt_seconds, 0.0],
        [0.0, 0.0, average_dt_seconds]
    ], dtype=float)

    pos_history = []
    for row in df_data.itertuples(index=False):
        u = numpy.array((row.x, row.y, row.z), dtype=float)
        kfilter.predict(u=u)

        pos_history.append(kfilter.x[3:6])
    pos_history = numpy.vstack(pos_history)

    fig, axes = plt.subplots(3)
    axes[0].set_title("x vs y")
    axes[0].scatter(pos_history[:, 0], pos_history[:, 1], marker="+")

    axes[1].set_title("x vs z")
    axes[1].scatter(pos_history[:, 0], pos_history[:, 2], marker="+")

    axes[2].set_title("y vs z")
    axes[2].scatter(pos_history[:, 1], pos_history[:, 2], marker="+")

    fig.tight_layout(pad=1.0)

    plt.show()

if __name__ == "__main__":
    main()