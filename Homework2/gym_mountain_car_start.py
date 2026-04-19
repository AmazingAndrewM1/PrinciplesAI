import argparse
import gymnasium as gym
import pickle
import random


def modifyReward(state, next_state, action, reward):
    """
    Modify the reward so that your DQN policy search succeeds.
    """
    # This is part of your homework.
    # You'll probably want to reward the model for getting closer to the goal, somehow.
    pass


def discretizeState(state):
    """
    Discretize the state. Used with value iteration.
    """
    # This is part of your homework
    pass

if __name__ == "__main__":
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

    max_steps = 4000
    env = gym.make('MountainCarContinuous-v0', max_episode_steps=max_steps)

    # We can go up, down, left, or right
    print(f"Action space: {env.action_space}")
    # We know the vehicle position and speed
    print(f"Observation space: {env.observation_space}")

    # The discretized, value-iteration solution does not need training.
    if args.dnn != 'discretized':
        for episode in range(args.episodes):
            # Reset the environment to put the agent into an initial state
            state, info = env.reset()

            # Run the simulation
            finished = False

            sim_steps = 0
            while not(finished):
                # TODO Get an action from your model
                action = [0]

                next_state, reward, terminated, truncated, info = env.step(action)

                if args.dnn == 'dqn':
                    reward = modifyReward(state, next_state, action, reward)

                # Perform an update step for your deep Q network.
                # TODO

                if terminated:
                    next_state = None

                # Update the state
                state = next_state

                finished = terminated or truncated
    env.close()

    # Play with policy
    if args.record:
        env = gym.make('MountainCarContinuous-v0', max_episode_steps=4000, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            #episode_trigger=lambda num: num % 2 == 0,
            video_folder="./",
            name_prefix="mountain-car",
        )
    else:
        env = gym.make('MountainCarContinuous-v0', max_episode_steps=4000, render_mode="human")
    state, info = env.reset()
    # Run the simulation
    finished = False

    while not(finished):
        if args.dnn == 'discretized':
            # The discretized model should not require learning, converging instead through value iteration.
            state = discretizeState(state)
            next_state = discretizeState(next_state)
            # TODO Get the next action from the discretized model
            action = [0]
        else:
            # TODO Get the next action from your DQN
            action = [0]

        # Take the action
        #print(f"Taking action {action} from state {state}")
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update the state
        state = next_state

        finished = terminated or truncated
    env.close()

