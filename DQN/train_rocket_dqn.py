# train_rocket_dqn.py
import torch
import numpy as np
from rocket_env import RocketEnv
from dueling_dqn_per import DQNAgent

def train_dqn(
    num_episodes=1000,
    max_steps=1000,
    render_interval=50,
    device="cuda"  # or "cpu"
):
    env = RocketEnv(max_steps=max_steps, task="landing", wind=False, wind_scale=2.0)
    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n            # 9
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=100000,
        min_buffer=2000,             # wait until 2k transitions before training
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=200000,        # steps for epsilon to decay
        target_update_freq=1000,
        alpha=0.6, 
        beta_start=0.4,
        beta_frames=200000,
        device=device
    )
    
    episode_rewards = []
    
    total_steps = 0
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        for step in range(max_steps):
            # Optionally render every N episodes for visualization
            if render_interval > 0 and ep % render_interval == 0:
                env.render()
            
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # DQN update
            agent.update()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep} | Steps: {step} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Save final model
    agent.save("rocket_dqn_checkpoint.pth")
    env.close()
    return episode_rewards

if __name__ == "__main__":
    rewards = train_dqn(num_episodes=6000000, max_steps=1000, render_interval=50, device="cpu")
    # Plot or analyze rewards as needed
