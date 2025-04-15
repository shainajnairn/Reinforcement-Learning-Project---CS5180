# train_rocket_dqn.py
import torch
import numpy as np
from rocket_env import RocketEnv
from dueling_dqn_per import DQNAgent
from datetime import datetime
import pickle


def train_dqn(num_episodes=1000, max_steps=1000, render_interval=50, device="cuda"):
    env = RocketEnv(max_steps=max_steps, task="landing", wind=False, wind_scale=2.0)
    state_dim = env.observation_space.shape[0]  # e.g., 8
    action_dim = env.action_space.n  # e.g., 9

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=100000,
        min_buffer=2000,  # wait until 2k transitions before training
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=200000,  # steps for epsilon to decay
        target_update_freq=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=200000,
        device=device,
    )

    episode_rewards = []
    episode_lengths = []
    best_reward = float("-inf")
    best_reward_episode = None
    best_reward_total_steps = None
    total_steps = 0

    # Track start time
    start_time = datetime.now()

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps_in_episode = 0

        for step in range(max_steps):
            if render_interval > 0 and ep % render_interval == 0:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1
            steps_in_episode += 1

            # DQN update
            agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps_in_episode)

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_reward_episode = ep
            best_reward_total_steps = total_steps

        print(
            f"Episode {ep} | Steps: {steps_in_episode} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}"
        )

    # Save final model checkpoint
    agent.save("rocket_dqn_checkpoint.pth")
    env.close()

    # Track end time and compute training duration
    end_time = datetime.now()
    training_duration = end_time - start_time

    overall_avg_reward = (
        sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    )

    # Compute moving average reward with a window size of 10
    window_size = 10
    moving_avg_rewards = []
    for i in range(len(episode_rewards)):
        start_index = max(0, i - window_size + 1)
        window = episode_rewards[start_index : i + 1]
        moving_avg_rewards.append(sum(window) / len(window))

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "training_duration": training_duration,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "best_reward": best_reward,
        "best_reward_episode": best_reward_episode,
        "best_reward_total_steps": best_reward_total_steps,
        "total_episodes": num_episodes,
        "final_timestep": total_steps,
        "average_reward": overall_avg_reward,
        "moving_average_reward": moving_avg_rewards,
    }

    return metrics


if __name__ == "__main__":
    metrics_dqn = train_dqn(
        num_episodes=5000, max_steps=1000, render_interval=50, device="cpu"
    )
    with open("../saved_metrics/metrics_dqn.pkl", "wb") as f:
        pickle.dump(metrics_dqn, f)
