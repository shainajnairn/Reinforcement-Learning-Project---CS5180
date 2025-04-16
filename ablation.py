import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys

sys.path.insert(0, "PPO_agents")
from datetime import datetime

# from PPO import PPO
# from PPO_MINI import PPO as PPO_MINI
# from PPO_MINI_GAE import PPO as PPO_MINI_GAE
from PPO_GAE import PPO as PPO_GAE
from SAFE_PPO import PPO as PPO_SAFE

from rocket import Rocket

from tqdm.auto import tqdm
import gymnasium as gym

import pickle
import math


max_ep_len = 1000
env_name = "RocketLanding"
task = "landing"
# Initialize Rocket environment
env = Rocket(
    max_steps=max_ep_len, task=task, rocket_type="starship", wind=False, wind_scale=2.0
)


def train(
    agent,
    env,
    env_name="RocketLanding",
    task="landing",
    max_training_timesteps=6_000_000,
    render=False,
    max_ep_len=1000,
    print_freq=None,
    log_freq=None,
    save_model_freq=100_000,
    update_timestep=None,
    random_seed=0,
    convergence_threshold=None,
):
    """
    Train an agent on the Rocket environment.
    Returns training metrics, including episode rewards and average rewards.
    """
    # print(agent.__class__.__name__)

    if print_freq is None:
        print_freq = max_ep_len * 10  # e.g. 10000
    if log_freq is None:
        log_freq = max_ep_len * 2  # e.g. 2000
    if update_timestep is None:
        update_timestep = max_ep_len * 4  # e.g. 4000

    # Setup logging directory
    log_dir = "PPO_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, env_name)
    os.makedirs(log_dir, exist_ok=True)

    run_num = len(next(os.walk(log_dir))[2])  # counting files for naming
    log_f_name = os.path.join(log_dir, f"PPO_{env_name}_log_{run_num}.csv")
    print("Logging at :", log_f_name)

    # Setup checkpoint directory
    directory = "PPO_preTrained"
    os.makedirs(directory, exist_ok=True)
    directory = os.path.join(directory, env_name)
    os.makedirs(directory, exist_ok=True)

    checkpoint_path = os.path.join(
        directory, f"PPO_{env_name}_{random_seed}_{run_num}.pth"
    )
    print("Save checkpoint path :", checkpoint_path)

    # Track time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # Open log file
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    # tracking variables
    print_running_reward = 0.0
    print_running_episodes = 0
    log_running_reward = 0.0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # Storing metrics
    episode_rewards = []  # Episode-level rewards
    episode_lengths = []  # How many steps each episode took
    average_reward = []  # This will remain unused in favor of our new metrics
    if agent.name == "PPO_GAE":
        gae_returns_list = []

    best_reward = float("-inf")
    best_reward_episode = None
    best_reward_timestep = None

    # If we define a threshold for "convergence," we'll track when it was first reached
    convergence_timestep = None
    convergence_episode = None

    window_size = 10  # for smoothing the reward plot

    # Main training loop
    pbar = tqdm(total=max_training_timesteps, desc="Training Timesteps")
    agent.set_lr(lr_actor_start, lr_critic_start)

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0.0

        for t in range(1, max_ep_len + 1):
            # Select action
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # Log to PPO buffer
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            pbar.update(1)

            # Render if needed
            if render and i_episode % 50 == 0:
                env.render()

            # Update PPO
            if time_step % update_timestep == 0:
                if agent.name == "PPO_GAE":
                    avg_gae_return = agent.update()
                    gae_returns_list.append(avg_gae_return)
                else:
                    agent.update()

            # Logging to file
            if time_step % log_freq == 0:
                if log_running_episodes > 0:
                    log_avg_reward = log_running_reward / log_running_episodes
                else:
                    log_avg_reward = 0
                log_f.write(
                    "{},{},{}\n".format(i_episode, time_step, round(log_avg_reward, 4))
                )
                log_running_reward, log_running_episodes = 0.0, 0

            # Print average reward to console
            # if time_step % print_freq == 0:
            #     if print_running_episodes > 0:
            #         print_avg_reward = print_running_reward / print_running_episodes
            #     else:
            #         print_avg_reward = 0
            #     print(
            #         f"Episode : {i_episode} \t\t"
            #         f"Timestep : {time_step} \t\t"
            #         f"Average Reward : {round(print_avg_reward, 2)}"
            #     )
            #     print_running_reward, print_running_episodes = 0.0, 0

            # # Save model checkpoint
            if time_step % save_model_freq == 0:
                agent.save(checkpoint_path)
                print("Model saved at timestep:", time_step)

            if done:
                break

        # End of episode updates
        i_episode += 1
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        episode_rewards.append(current_ep_reward)
        episode_lengths.append(t)

        # Track best episode reward
        if current_ep_reward > best_reward:
            best_reward = current_ep_reward
            best_reward_episode = i_episode
            best_reward_timestep = time_step

        # Check if we reached a "convergence" threshold
        if (convergence_threshold is not None) and (convergence_timestep is None):
            if current_ep_reward >= convergence_threshold:
                convergence_timestep = time_step
                convergence_episode = i_episode

    pbar.close()
    log_f.close()

    end_time = datetime.now().replace(microsecond=0)
    print("Finished training at : ", end_time)
    print("Total training time  : ", end_time - start_time)

    # Calculate overall average reward across episodes
    if episode_rewards:
        overall_avg_reward = sum(episode_rewards) / len(episode_rewards)
    else:
        overall_avg_reward = 0

    # Compute moving average of episode rewards using the specified window size
    moving_avg_rewards = []
    for i in range(len(episode_rewards)):
        start_index = max(0, i - window_size + 1)
        window = episode_rewards[start_index : i + 1]
        moving_avg_rewards.append(sum(window) / len(window))

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "training_duration": end_time - start_time,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "best_reward": best_reward,
        "best_reward_episode": best_reward_episode,
        "best_reward_timestep": best_reward_timestep,
        "convergence_threshold": convergence_threshold,
        "convergence_timestep": convergence_timestep,
        "convergence_episode": convergence_episode,
        "total_episodes": i_episode,
        "final_timestep": time_step,
        "average_reward": overall_avg_reward,
        "moving_average_reward": moving_avg_rewards,
    }

    return metrics


if __name__ == "__main__":
    # State and action dimensions
    state_dim = env.state_dims
    action_dim = env.action_dims

    # gamma = 0.96
    # eps_clip = 0.1
    # K_epochs = 70
    # lr_actor = 0.00010154485581168334
    # lr_critic = 0.0006998820449991889
    # lam = 0.9
    # has_continuous_action_space = False
    lr_actor = 0.0003
    lr_critic = 0.001
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    has_continuous_action_space = False
    mini_batch_size = 64
    lam = 0.95

    ppo_gae_agent = PPO_GAE(
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

    ppo_safe_agent = PPO_SAFE(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        cost_gamma=0.99,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        has_continuous_action_space=has_continuous_action_space,
    )

    metrics_ppo_safe = train(
        ppo_safe_agent,
        env,
        env_name="RocketLanding",
        task="landing",
        max_training_timesteps=6e6,
        convergence_threshold=200,
    )

    with open("saved_metrics/metrics_ppo_safe.pkl", "wb") as f:
        pickle.dump(metrics_ppo_safe, f)

    metrics_ppo_gae = train(
        ppo_gae_agent,
        env,
        env_name="RocketLanding",
        task="landing",
        max_training_timesteps=6e6,
        convergence_threshold=200,
    )

    with open("saved_metrics/metrics_ppo_gae.pkl", "wb") as f:
        pickle.dump(metrics_ppo_gae, f)
