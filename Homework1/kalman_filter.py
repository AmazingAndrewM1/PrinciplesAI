import numpy
from matplotlib import pyplot as plt
import pandas as pd

class KalmanFilter:
    def __init__(self, dt, system_noise, measurement_noise):
        # State vector/mean estimate holds x,y,z acceleration, velocity, and position
        self.x = numpy.zeros((9, 1))

        # Transition matrix
        self.A = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0], # Acceleration comes from itself
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [dt, 0, 0, 1, 0, 0, 0, 0, 0], # Velocity is the current estimate plus the time delta * accel
            [0, dt, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, dt, 0, 0, 1, 0, 0, 0],
            [dt**2/2, 0, 0, dt, 0, 0, 1, 0, 0], # x, y, z location are the current estimate + dt * velocity + dt^2 * accel/2
            [0, dt**2/2, 0, 0, dt, 0, 0, 1, 0],
            [0, 0, dt**2/2, 0, 0, dt, 0, 0, 1]
        ])

        # Our observation matrix is only the acceleration
        self.C = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])
        # Make an observation matrix that contains the velocity for 0 velocity updates.
        self.zeroC = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        # Make an observation matrix that contains the location for location updates.
        self.fuseLocation = numpy.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Initial uncertainty (Sigma, the covariance matrix)
        # Notice that this is initialized to a diagonal matrix, but the
        # variance update will project P by the transition matrix when
        # we do this: self.C @ self.P @ self.C.T
        self.P = numpy.eye(9)
        # System noise (sigma in the equations)
        self.Q = numpy.eye(9) * system_noise
        # Measurement noise (delta in the equations)
        self.R = numpy.ones((3,1)) * measurement_noise
        # Measurement noise used during a zero velocity update
        self.jitter = numpy.ones((6, 1)) * 10e-5
        # Measurement noise used during sensor fusion
        self.fusion_jitter = numpy.ones((9, 1)) * 10e-5

    def predict(self):
        # Update mean and variance predictions using the transition matrix
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def control(self, B, u):
        # Call after predict to update with a control input
        # If the control matrix exists we follow the full equation:
        #  \mu_{t|t-1} \triangleeq A_t\mu{t-1} + B_t u_t
        self.x += B @ u

    def _internal_update(self, C, y, measure_noise):
        # Compare the prediction to the observed state
        y_hat = C @ self.x
        residual = y - y_hat
        # Update the variance
        S = C @ self.P @ C.T + measure_noise
        # Kalman gain matrix
        K = self.P @ C.T @ numpy.linalg.inv(S)

        # Update from the mean and covariance. 18.31-18.32 in Murphy's book.
        self.x = self.x + K @ residual
        self.P = (numpy.eye(len(self.P)) - K @ C) @ self.P

    def zero_velocity_update(self):
        # As an alternative to the control input, we can also expand the
        # observation to include velocity, and indicate that it is 0.
        # This doesn't interrupt the statistics of the system, so it could be preferable.
        y = numpy.zeros((6, 1))
        self._internal_update(self.zeroC, y, self.jitter)

    def fused_location_update(self, y_accel, y_location):
        # As an alternative to the control input, we can also expand the
        # observation to include location.
        # This doesn't interrupt the statistics of the system, so it could be preferable.
        y = numpy.concatenate((y_accel.reshape(-1, 1), self.x[3:6], y_location.reshape(-1, 1)), axis=0)
        self._internal_update(self.fuseLocation, y, self.fusion_jitter)

    def update(self, y):
        """
        Arguments:
            y: y is the observed state.
        """
        y = y.reshape(-1, 1)
        self._internal_update(self.C, y, self.R) 

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

def get_approximate_measurement_noise():
    STILL_DATA_FILE_PATH = "./still-data/Accelerometer.csv"
    df_still_data = get_trimmed_data(STILL_DATA_FILE_PATH)
    return df_still_data.loc[:, ["x", "y", "z"]].cov().to_numpy()

def main():
    DATA_FILE_PATH = "./data/Accelerometer.csv"
    df_data = get_trimmed_data(DATA_FILE_PATH)

    average_dt_seconds = get_average_dt_seconds(df_data["time"])
    system_noise = get_approximate_system_noise(0.001, average_dt_seconds)
    measurement_noise = get_approximate_measurement_noise()
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

    plt.plot(df_filtered.loc[:, "pos_x"], df_filtered.loc[:, "pos_z"], marker=",", label="Estimated position")
    plt.show()

if __name__ == "__main__":
    main()