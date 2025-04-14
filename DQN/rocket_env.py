# rocket_env.py
import numpy as np
import gym
from gym import spaces
import random
from rocket import Rocket

# If your Rocket class is in the same file, you can move it above or do:
# from rocket import Rocket

class RocketEnv(gym.Env):
    """
    A Gym-like interface for your discrete rocket environment.
    """
    def __init__(
        self,
        max_steps=1000,
        task="landing", 
        rocket_type="starship",
        wind=True,
        wind_scale=2.0,
    ):
        super(RocketEnv, self).__init__()
        self.max_steps = max_steps
        
        # Instantiate your existing Rocket environment
        self.rocket = Rocket()
        self.rocket._init_(
            max_steps=max_steps,
            task=task,
            rocket_type=rocket_type,
            wind=wind,
            wind_scale=wind_scale
        )
        
        # Discrete action space: 9 possible actions
        self.action_space = spaces.Discrete(self.rocket.action_dims)
        
        # Observation: 8-dimensional, each normalized by 100 in .flatten()
        # We'll assume each dimension is in range roughly [-5, +5] after normalization
        # You can refine these bounds based on domain knowledge.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(8,), dtype=np.float32
        )
        
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        obs = self.rocket.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.rocket.step(action)
        self.current_step += 1
        # Enforce max_steps
        if self.current_step >= self.max_steps:
            done = True
        return obs, reward, done, info

    def render(self, mode="human"):
        # Show OpenCV window
        self.rocket.render(wait_time=1)
        
    def close(self):
        pass