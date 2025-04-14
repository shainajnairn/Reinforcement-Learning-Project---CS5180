# dueling_dqn_per.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------------------------
# 1) Prioritized Experience Replay Buffer
# -------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        
        # For priority sampling
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.eps = 1e-5  # small constant to avoid 0 priority

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio if max_prio > 0 else self.eps
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # If buffer not full, only use existing data
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        
        # Convert priorities -> probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize
        
        # Separate out each component
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(prio, self.eps)

    def __len__(self):
        return len(self.buffer)


# -------------------------------------------------
# 2) Dueling Network Architecture
# -------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.hidden(x)
        value = self.value_stream(features)              # [batch_size, 1]
        advantages = self.advantage_stream(features)     # [batch_size, action_dim]
        # Q = V(s) + A(s,a) - mean(A(s,:))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


# -------------------------------------------------
# 3) DQNAgent: uses Double DQN + PER + Dueling net
# -------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=100000,
        min_buffer=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=100000,
        target_update_freq=1000,
        alpha=0.6,   # PER alpha
        beta_start=0.4,
        beta_frames=100000,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        self.device = device
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Networks: policy & target
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.target_update_freq = target_update_freq
        self.learn_step = 0  # to track when to update target

    def select_action(self, state):
        """
        Epsilon-greedy action selection:
        state: np array of shape (state_dim,)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]
            with torch.no_grad():
                q_values = self.policy_net(state_t)
                action = q_values.argmax(dim=1).item()
            return action

    def push_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        One gradient update of the DQN if buffer is large enough.
        Uses Double DQN and PER.
        """
        if len(self.buffer) < self.min_buffer:
            return
        
        # Linear annealing of epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        )
        
        # Beta for importance sampling (linearly annealed)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        (states, actions, rewards, next_states, dones,
         indices, weights) = self.buffer.sample(self.batch_size, beta=beta)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)         # [batch_size, state_dim]
        actions_t = torch.LongTensor(actions).to(self.device)        # [batch_size]
        rewards_t = torch.FloatTensor(rewards).to(self.device)       # [batch_size]
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)       # [batch_size]
        
        # Current Q
        q_values = self.policy_net(states_t)                         # [batch_size, action_dim]
        q_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Double DQN: next action from policy_net, Q from target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q_values * (1 - dones_t)
        
        # TD-errors
        td_errors = q_action - target
        loss = (weights_t * F.smooth_l1_loss(q_action, target, reduction='none')).mean()
        
        # Update priorities
        new_priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.buffer.update_priorities(indices, new_priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.learn_step += 1
        # Update target net
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
