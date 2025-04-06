import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import utils

device = utils.get_device()


################################## PPO Policy with minibatch ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self, state_dim, action_dim, has_continuous_action_space, action_std_init
    ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            ).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std
            ).to(device)
        else:
            print(
                "WARNING: Calling ActorCritic::set_action_std() on discrete action space policy"
            )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
        mini_batch_size=64,  # Mini-batch size for update
        lam=0.95,  # GAE lambda
    ):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lam = lam  # For GAE

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.mini_batch_size = mini_batch_size

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print(
                "WARNING: Calling PPO::set_action_std() on discrete action space policy"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std -= action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print(
                    "setting actor output action_std to min_action_std:",
                    self.action_std,
                )
            else:
                print("setting actor output action_std to:", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print(
                "WARNING: Calling PPO::decay_action_std() on discrete action space policy"
            )

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def update(self):
        """
        PPO update using GAE(λ) for advantage estimation and mini-batch updates.
        """
        # Convert list data to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(
            torch.stack(self.buffer.state_values, dim=0)
        ).detach()

        rewards = self.buffer.rewards
        dones = self.buffer.is_terminals

        # Compute GAE advantages
        advantages = []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[i] + self.gamma * next_value - old_state_values[i].item()
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            next_value = old_state_values[i].item()

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + old_state_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = old_states.size(0)
        indices = torch.arange(dataset_size)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Shuffle indices at the beginning of each epoch
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = perm[start:end]

                mb_states = old_states[mb_idx].to(device)
                mb_actions = old_actions[mb_idx].to(device)
                mb_logprobs = old_logprobs[mb_idx].to(device)
                mb_advantages = advantages[mb_idx].to(device)
                mb_returns = returns[mb_idx].to(device)

                # Evaluate current policy on mini-batch
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mb_states, mb_actions
                )
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - mb_logprobs)

                # Calculate surrogate losses
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mb_advantages
                )

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, mb_returns)
                    - 0.01 * dist_entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        # Update old policy and clear buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
