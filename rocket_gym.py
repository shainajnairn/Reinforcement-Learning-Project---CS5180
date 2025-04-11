import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rocket import Rocket
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


class RewardLandingTracker(BaseCallback):
    """
    Custom callback for plotting average reward and success landings after training.
    """

    def __init__(self, window_size=10, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_successes = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        """
        This is called at each environment step.
        """
        # 'self.locals["rewards"]' is a vector of rewards for each env in the vec env
        # 'self.locals["dones"]'   is a boolean array of done flags
        # 'self.locals["infos"]'   is an array of info dicts
        for i, done in enumerate(self.locals["dones"]):
            # Add the immediate reward from this step
            self.current_reward += self.locals["rewards"][i]

            if done:
                info = self.locals["infos"][i]
                # Episode finished => record reward
                self.episode_rewards.append(self.current_reward)
                # Check success
                is_success = info.get("is_success", False)
                self.episode_successes.append(1 if is_success else 0)

                # Reset for next episode
                self.current_reward = 0.0

        return True

    def _on_training_end(self) -> None:
        """
        Called at the very end of training.
        We'll plot the rolling average of rewards and total successes over episodes.
        """
        episodes = np.arange(1, len(self.episode_rewards) + 1)

        # Compute a simple moving average of the episode rewards
        avg_rewards = []
        for i in range(len(self.episode_rewards)):
            start = max(0, i - self.window_size + 1)
            avg_rewards.append(np.mean(self.episode_rewards[start : i + 1]))

        # 1) Plot the moving average rewards
        plt.figure()
        plt.plot(episodes, avg_rewards)
        plt.title("Moving Average Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

        # 2) Plot the cumulative number of successful landings
        success_cumsum = np.cumsum(self.episode_successes)
        plt.figure()
        plt.plot(episodes, success_cumsum)
        plt.title("Number of Successful Landings")
        plt.xlabel("Episode")
        plt.ylabel("Success Count")
        plt.show()


class RocketImageEnv(gym.Env):
    """
    A Gymnasium-compatible wrapper that returns raw image frames from
    the custom Rocket environment as observations.
    """

    def __init__(
        self,
        max_steps=500,
        task="hover",
        rocket_type="falcon",
        wind=False,
        viewport_h=768,
    ):
        """
        Args:
            max_steps:  Maximum number of steps before truncation
            task:       "hover" or "landing"
            rocket_type:"falcon" or "starship"
            wind:       Whether wind is enabled
            viewport_h: Height of the rendering viewport in pixels.
                        The width is automatically scaled by the Rocket env.
        """
        super().__init__()

        # Create the underlying Rocket environment
        self.env = Rocket(
            max_steps=max_steps,
            task=task,
            rocket_type=rocket_type,
            viewport_h=viewport_h,
            wind=wind,
        )

        self.max_steps = max_steps
        self.current_step = 0

        # ----------------------------------------------
        # 1) Define discrete action space of size 9
        #    from env.create_action_table().
        # ----------------------------------------------
        self.action_space = spaces.Discrete(self.env.action_dims)

        # ----------------------------------------------
        # 2) Define the image-based observation space
        #
        # The Rocket env uses:
        #   self.viewport_h = viewport_h
        #   self.viewport_w = ...
        # For the color channels, we assume 3 (RGB).
        # Values are in [0..255], so dtype=uint8
        # ----------------------------------------------
        obs_height = self.env.viewport_h
        obs_width = self.env.viewport_w
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        """Gymnasium API: resets the episode, returns (obs, info)."""
        super().reset(seed=seed)
        self.current_step = 0

        # Reset the underlying rocket environment
        _ = self.env.reset()

        # Render initial frame
        frame_0, _ = self.env.render(
            window_name="env",
            wait_time=1,
            with_trajectory=False,  # turn off if it distracts the agent
            with_camera_tracking=False,
        )

        # Ensure shape matches (H, W, 3)
        obs = frame_0  # or do any post-processing you like
        info = {}
        return obs, info

    def step(self, action):
        """
        Gymnasium API: does one env step.
        Must return (obs, reward, terminated, truncated, info).
        """
        self.current_step += 1

        # Step the rocket environment
        _, reward, done, info = self.env.step(action)

        # In Gymnasium, we split 'done' into 'terminated' vs 'truncated'
        terminated = done
        truncated = self.current_step >= self.max_steps

        # Render the new frame as the next observation
        frame_0, _ = self.env.render(
            window_name="env",
            wait_time=1,
            with_trajectory=False,
            with_camera_tracking=False,
        )
        obs = frame_0

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        (Optional) If you want manual rendering calls outside step().
        We'll just reuse the rocket's internal render here.
        """
        return self.env.render()


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# 1) Create environment
env = RocketImageEnv(
    max_steps=500, task="hover", rocket_type="falcon", wind=False, viewport_h=128
)

# 2) Create evaluation environment
eval_env = RocketImageEnv(
    max_steps=500, task="hover", rocket_type="falcon", wind=False, viewport_h=128
)

# 3) Create PPO model
policy_kwargs = dict(
    net_arch=[256, 256, 128], activation_fn=nn.ReLU  # Three hidden layers
)

log_dir = "./tensorboard"

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    learning_rate=3e-4,
    tensorboard_log=log_dir,  # Where to log for TensorBoard
    policy_kwargs=policy_kwargs,  # Pass in our custom architecture
)
# 4) Create a reward-logging callback
reward_tracker = RewardLandingTracker(window_size=10)

# Optionally, an eval callback to periodically evaluate on eval_env
eval_callback = EvalCallback(eval_env, eval_freq=5000, deterministic=True, render=False)

# 5) Train with callbacks
model.learn(total_timesteps=500_000, callback=[reward_tracker, eval_callback])

# 6) Save final model
model.save("rocket_ppo_model")

obs, _ = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
