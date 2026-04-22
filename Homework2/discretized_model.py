# Source code of Continuous Mountain Car Env
# https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/continuous_mountain_car.py

from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import numpy as np

class DiscretizedModel:
    def __init__(self, unwrapped_env: Continuous_MountainCarEnv):
        self.unwrapped_env = unwrapped_env

        self.pos_bins = np.linspace(unwrapped_env.min_position, unwrapped_env.max_position, endpoint=True, dtype=float)
        self.vel_bins = np.linspace(-unwrapped_env.max_speed, unwrapped_env.max_speed, endpoint=True, dtype=float)
        self.action_bins = np.array([unwrapped_env.min_action, 0.0, unwrapped_env.max_action], dtype=float)

        self.values = np.zeros((self.pos_bins.shape[0], self.vel_bins.shape[0]), dtype=float)

    def to_discrete(self, continuous_state):
        pos, vel = continuous_state
        
        pos_clipped = np.clip(pos, self.unwrapped_env.min_position, self.unwrapped_env.max_position)
        pos_i = np.digitize(pos_clipped, self.pos_bins) - 1

        vel_clipped = np.clip(vel, -self.unwrapped_env.max_speed, self.unwrapped_env.max_speed)
        vel_i = np.digitize(vel_clipped, self.vel_bins) - 1

        return pos_i, vel_i
    
    def to_continuous(self, discrete_state):
        pos_i, vel_i = discrete_state
        return self.pos_bins[pos_i], self.vel_bins[vel_i]
    
    def transition(self, discrete_state, action):
        pos, vel = self.to_continuous(discrete_state)

        # Simulate exactly as done in the simulation. I know this is right because I cross-referenced the source code.
        next_vel: float = np.clip(
            vel + action * self.unwrapped_env.power - 0.0025 * np.cos(3.0 * pos), 
            -self.unwrapped_env.max_speed, 
            self.unwrapped_env.max_speed
        )
        next_pos: float = np.clip(
            pos + next_vel,
            self.unwrapped_env.min_position,
            self.unwrapped_env.max_position
        )

        if (next_pos == self.unwrapped_env.min_position and next_vel < 0.0) or (next_pos == self.unwrapped_env.max_position and next_vel > 0.0):
            next_vel = 0.0

        reward = 1.0 if next_pos > self.unwrapped_env.goal_position else 0.0

        return (next_pos, next_vel), reward




