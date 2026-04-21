import argparse
import gymnasium as gym
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import pickle
import random
from agent import Agent
from discretized_model import DiscretizedModel
from matplotlib import pyplot as plt
import tqdm

has_printed = False

def modifyReward(state, next_state, action, reward, env: Continuous_MountainCarEnv):
    """
    Modify the reward so that your DQN policy search succeeds.
    """
    # This is part of your homework.
    # You'll probably want to reward the model for getting closer to the goal, somehow.

    # I thought about mechanical energy = kinetic energy + potential energy from physics to create this reward function
    curr_position, curr_velocity = state
    mechanical_energy_current = 0.5 * curr_velocity * curr_velocity + env._height(curr_position)

    next_position, next_velocity = next_state
    mechanical_energy_next = 0.5 * next_velocity * next_velocity + env._height(next_position)
    return mechanical_energy_next - mechanical_energy_current

def discretizeState(state):
    """
    Discretize the state. Used with value iteration.
    """
    # This is part of your homework
    pass

if __name__ == "__main__":
    # Start simulation!
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=False,
        type=str,
        default='dnn',
        choices=['dqn', 'discretized'],
        help="Policy search algorithm.")
    parser.add_argument(
        "--record",
        required=False,
        default=False,
        action='store_true',
        help="Record the end result as a video.")
    parser.add_argument(
        "--episodes",
        required=False,
        type=int,
        default=100,
        help="The number of episodes to run")
    args = parser.parse_args()

    max_steps = 1_000
    env: gym.wrappers = gym.make('MountainCarContinuous-v0', max_episode_steps=max_steps)

    # We can go up, down, left, or right
    print(f"Action space: {env.action_space}")
    # We know the vehicle position and speed
    print(f"Observation space: {env.observation_space}")

    # Environment information

    agent = Agent(env)

    # The discretized, value-iteration solution does not need training.
    if args.algorithm != 'discretized':
        for episode in tqdm.trange(args.episodes):
            # Reset the environment to put the agent into an initial state
            state, info = env.reset()

            # Run the simulation
            finished = False
            sim_steps = 0
            while not(finished):
                # TODO Get an action from your model
                action_index = agent.choose_action_index(state)
                action = agent.choose_action_from_index(action_index)

                next_state, reward, terminated, truncated, info = env.step(action)

                if args.algorithm == 'dqn':
                    reward = modifyReward(state, next_state, action, reward, env.unwrapped)

                # Perform an update step for your deep Q network.
                # TODO
                agent.decay_epsilon()
                agent.replay_buffer.add(state, action_index, reward, next_state, terminated or truncated)
                agent.train()

                if terminated:
                    next_state = None

                # Update the state
                state = next_state

                finished = terminated or truncated

        plt.plot(agent.loss_history)
        plt.savefig("./plots/loss_history.png", format="png", bbox_inches="tight")
    env.close()

    # Play with policy
    if args.record:
        env = gym.make('MountainCarContinuous-v0', max_episode_steps=4000, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            #episode_trigger=lambda num: num % 2 == 0,
            video_folder="./videos/",
            name_prefix="mountain-car",
        )
    else:
        env = gym.make('MountainCarContinuous-v0', max_episode_steps=4000, render_mode="human")
    state, info = env.reset()
    # Run the simulation
    finished = False

    num_steps = 0

    if args.algorithm == "discretized":
        discretized_model = DiscretizedModel(env.unwrapped)

    while not(finished):
        if args.algorithm == 'discretized':
            # The discretized model should not require learning, converging instead through value iteration.
            state = discretizeState(state)
            next_state = discretizeState(next_state)
            # TODO Get the next action from the discretized model
            action = [0]
        else:
            # TODO Get the next action from your DQN
            action_index = agent.choose_action_index(state)
            action = agent.choose_action_from_index(action_index)

        # Take the action
        #print(f"Taking action {action} from state {state}")
        next_state, reward, terminated, truncated, info = env.step(action)
        num_steps += 1

        # Update the state
        state = next_state

        finished = terminated or truncated
    env.close()
    print(f"Num Steps to Solution: {num_steps}")
