from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import numpy as np

class DiscretizedModel:
    def __init__(self, unwrapped_env: Continuous_MountainCarEnv):
        self.pos_bins = np.linspace(unwrapped_env.min_position, unwrapped_env.max_position, endpoint=True)
        self.vel_bins = np.linspace(-unwrapped_env.max_speed, unwrapped_env.max_speed, endpoint=True)
