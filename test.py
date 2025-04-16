import os
import time
from datetime import datetime

import torch
import numpy as np
import sys

sys.path.insert(0, "PPO_agents")
from SAFE_PPO import PPO as PPO_SAFE
from PPO_GAE import PPO as PPO_GAE

from rocket import Rocket


#################################### Testing ###################################
def test():
    ################## Hyperparameters ##################
    env_name = "RocketLanding"
    task = "landing"  # 'hover' or 'landing'

    has_continuous_action_space = False
    max_ep_len = 1000  # Max timesteps in one episode

    render = True  # Render environment on screen
    frame_delay = 1  # Delay between frames (in seconds)

    total_test_episodes = 10  # Total number of testing episodes

    K_epochs = 80  # Update policy for K epochs
    eps_clip = 0.2  # Clip parameter for PPO
    gamma = 0.99  # Discount factor

    lr_actor = 0.0003  # Learning rate for actor
    lr_critic = 0.001  # Learning rate for critic
    lam = 0.95
    #####################################################

    # Initialize the Rocket environment
    env = Rocket(
        max_steps=max_ep_len, task=task, rocket_type="starship"
    )  # Adjust for 'hover' task if needed

    # Set state and action dimensions based on Rocket's configuration
    state_dim = env.state_dims
    action_dim = env.action_dims

    # Initialize a PPO agent

    # ppo_agent = PPO_SAFE(
    #     state_dim=state_dim,
    #     action_dim=action_dim,
    #     lr_actor=lr_actor,
    #     lr_critic=lr_critic,
    #     gamma=gamma,
    #     cost_gamma=0.99,
    #     K_epochs=K_epochs,
    #     eps_clip=eps_clip,
    #     has_continuous_action_space=has_continuous_action_space,
    # )

    ppo_agent = PPO_GAE(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        lam=lam,
    )

    # Pretrained weights directory
    random_seed = 0  # Set this to load a specific checkpoint trained on a random seed
    run_num_pretrained = 266 # Set this to load a specific checkpoint number

    directory = "./PPO_preTrained" + "/" + env_name + "/"
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
        env_name, random_seed, run_num_pretrained
    )
    # checkpoint_path = "./PPO_logs/RocketLanding/PPO_RocketLanding_log_264.csv"
    print("loading network from : " + checkpoint_path)

    # Load pretrained model
    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render(
                    window_name="Rocket Test", wait_time=frame_delay
                )  # Adjust for Rocket render method

            if done:
                break

        # Clear PPO agent buffer after each episode
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print("Episode: {} \t\t Reward: {}".format(ep, round(ep_reward, 2)))

    # env.close()

    avg_test_reward = test_running_reward / total_test_episodes
    print("average test reward : " + str(round(avg_test_reward, 2)))


if __name__ == "__main__":
    test()
