import numpy

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