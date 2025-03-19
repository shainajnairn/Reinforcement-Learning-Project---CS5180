import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from PPO import PPO

from PPO_MINI import PPO as PPO_MINI

# Import the Rocket environment
from rocket import Rocket


def train_simultaneously():
    """
    Train two PPO agents (original vs. mini-batch) side by side.
    They each get their own environment, but we step them together in one loop.
    """

    ############################################################################
    # 1) Environment and PPO configuration
    ############################################################################
    env_name = "RocketLanding"
    task = "landing"

    has_continuous_action_space = False  # Typically discrete for Rocket
    max_ep_len = 1000
    max_training_timesteps = int(3e5)  # reduce for demonstration

    # PPO hyperparameters
    update_timestep = max_ep_len * 4  # e.g. 4000
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 3e-4
    lr_critic = 1e-3

    # Intervals
    print_freq = max_ep_len * 5
    save_model_freq = int(1e5)

    # Create two separate environments
    env1 = Rocket(max_steps=max_ep_len, task=task, rocket_type="starship")
    env2 = Rocket(max_steps=max_ep_len, task=task, rocket_type="starship")

    # Dimensions
    state_dim = env1.state_dims
    action_dim = env1.action_dims

    # Create PPO agents
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
    )

    ppo2_agent = PPO_MINI(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        mini_batch_size=64,  # example
    )

    # Episode rewards
    episode_rewards_1 = []
    episode_rewards_2 = []

    # Current episode reward accumulators
    current_ep_reward_1 = 0.0
    current_ep_reward_2 = 0.0

    # Episode counters
    episode_count_1 = 0
    episode_count_2 = 0

    # Initialize states in each environment
    state1 = env1.reset()
    state2 = env2.reset()

    # Keep track of timesteps
    time_step = 0

    ############################################################################
    # 3) Setup real-time plot
    ############################################################################
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("PPO Training (Simultaneous): Original vs Mini-Batch")
    plt.show(block=False)

    window_size = 10  # for moving average

    def update_plot(ax, ep_rewards_1, ep_rewards_2):
        """
        Updates the single figure with two lines: PPO1 vs. PPO2,
        each with a moving-average and standard-deviation shading.
        """
        ax.clear()

        def plot_moving_avg_std(data, label):
            episodes = np.arange(len(data))
            if len(data) < window_size:
                # Not enough episodes for a meaningful moving avg -> just plot raw
                ax.plot(episodes, data, label=label)
            else:
                # Moving average
                mov_avg = np.convolve(data, np.ones(window_size) / window_size, "valid")
                # Moving std
                mov_std = []
                for i in range(window_size - 1, len(data)):
                    segment = data[i - window_size + 1 : i + 1]
                    mov_std.append(np.std(segment))
                mov_std = np.array(mov_std)

                x_vals = np.arange(window_size - 1, len(data))
                ax.plot(x_vals, mov_avg, label=label)
                ax.fill_between(x_vals, mov_avg - mov_std, mov_avg + mov_std, alpha=0.2)

        # Plot both agents
        plot_moving_avg_std(np.array(ep_rewards_1), "Original PPO")
        plot_moving_avg_std(np.array(ep_rewards_2), "Mini-Batch PPO")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("PPO Training (Simultaneous): Original vs Mini-Batch")
        ax.legend()
        plt.draw()
        plt.pause(0.01)

    ############################################################################
    # 4) Main training loop (side-by-side stepping)
    ############################################################################
    start_time = datetime.now().replace(microsecond=0)
    print(f"Started training at (GMT): {start_time}\n")

    while time_step <= max_training_timesteps:
        time_step += 1

        # -----------------------------
        # Agent1 step
        # -----------------------------
        action1 = ppo_agent.select_action(state1)
        next_state1, reward1, done1, _ = env1.step(action1)

        # Log to agent1's buffer
        ppo_agent.buffer.rewards.append(reward1)
        ppo_agent.buffer.is_terminals.append(done1)

        # Accumulate reward
        current_ep_reward_1 += reward1

        if done1:
            # Episode finished for agent1
            episode_rewards_1.append(current_ep_reward_1)
            episode_count_1 += 1
            current_ep_reward_1 = 0.0
            state1 = env1.reset()
        else:
            # Continue
            state1 = next_state1

        # -----------------------------
        # Agent2 step
        # -----------------------------
        action2 = ppo2_agent.select_action(state2)
        next_state2, reward2, done2, _ = env2.step(action2)

        # Log to agent2's buffer
        ppo2_agent.buffer.rewards.append(reward2)
        ppo2_agent.buffer.is_terminals.append(done2)

        # Accumulate reward
        current_ep_reward_2 += reward2

        if done2:
            # Episode finished for agent2
            episode_rewards_2.append(current_ep_reward_2)
            episode_count_2 += 1
            current_ep_reward_2 = 0.0
            state2 = env2.reset()
        else:
            state2 = next_state2

        # -----------------------------
        # PPO Updates
        # -----------------------------
        if time_step % update_timestep == 0:
            ppo_agent.update()
            ppo2_agent.update()

        # -----------------------------
        # Save model
        # -----------------------------
        if time_step % save_model_freq == 0:
            ppo_agent.save(checkpoint_path1)
            ppo2_agent.save(checkpoint_path2)
            print(f"[{time_step}] Models saved.")

        # -----------------------------
        # Print some info
        # -----------------------------
        if time_step % print_freq == 0:
            print(
                f"Time step: {time_step} | Episodes (Agent1={episode_count_1}, Agent2={episode_count_2})"
            )

        # -----------------------------
        # Real-time plotting
        # -----------------------------
        # update the plot whenever **either** agent completes an episode
        if done1 or done2:
            update_plot(ax, episode_rewards_1, episode_rewards_2)

    # End of training
    end_time = datetime.now().replace(microsecond=0)
    print(f"\nFinished training at (GMT): {end_time}")
    print(f"Training duration: {end_time - start_time}")

    plt.ioff()
    update_plot(ax, episode_rewards_1, episode_rewards_2)
    plt.show()


if __name__ == "__main__":
    train_simultaneously()
