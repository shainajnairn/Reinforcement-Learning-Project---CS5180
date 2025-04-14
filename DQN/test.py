# test_rocket_dqn.py
import torch
import numpy as np
from rocket_env import RocketEnv
from dueling_dqn_per import DQNAgent

def test_dqn(
    num_episodes=10,
    max_steps=500,
    checkpoint="rocket_dqn_checkpoint.pth",
    device="cpu",
    render=True
):
    """
    Runs the saved DQN model for a fixed number of episodes (default=10).
    """
    # Create the Rocket environment
    env = RocketEnv(
        max_steps=max_steps,
        task="landing",
        rocket_type="falcon",
        wind=True,
        wind_scale=2.0
    )
    
    state_dim = env.observation_space.shape[0]  # should be 8
    action_dim = env.action_space.n            # should be 9
    
    # Create a DQNAgent with the same architecture as the trained model
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Load the saved checkpoint
    agent.load(checkpoint)
    
    # Optionally set epsilon to a small value or zero for purely greedy actions
    agent.epsilon = 0.0  # purely greedy; or 0.05 if you want minimal exploration
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done and step_count < max_steps:
            if render:
                env.render()
                            
            # Choose the best action with the loaded policy (greedy if epsilon=0)
            action = agent.select_action(state)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1
        
        is_success = info.get("is_success", False)
        print(f"Episode {episode+1}: steps={step_count}, reward={episode_reward:.2f}, success={is_success}")
    
    env.close()

if __name__ == "__main__":
    test_dqn(
        num_episodes=10,
        max_steps=500,
        checkpoint="rocket_dqn_checkpoint.pth",  # Path to your saved model
        device="cpu",                            # or "cuda" if you have a GPU
        render=True
    )
