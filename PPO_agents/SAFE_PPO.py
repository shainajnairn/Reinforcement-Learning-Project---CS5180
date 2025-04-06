import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


#  ROLLOUT BUFFER
class SafeRolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.costs = []  # <--- new
        self.logprobs = []
        self.value_r = []  # predicted reward value
        self.value_c = []  # predicted cost value
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.costs[:]  # <--- new
        del self.logprobs[:]
        del self.value_r[:]
        del self.value_c[:]
        del self.is_terminals[:]


#  ACTOR-CRITIC NETWORK
class SafeActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SafeActorCritic, self).__init__()

        # For discrete actions, we have an actor that outputs a probability distribution
        hidden_size = 64

        # Actor (same as your normal PPO code)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic for reward
        self.critic_r = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Critic for cost
        self.critic_c = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self):
        # Not used directly; we use act() and evaluate().
        raise NotImplementedError

    def act(self, state):
        """
        Given a single state, returns:
          - action sampled from pi(.|state)
          - log prob of that action
          - reward value function for that state
          - cost  value function for that state
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        value_r = self.critic_r(state)
        value_c = self.critic_c(state)

        return action, logprob, value_r, value_c

    def evaluate(self, states, actions):
        """
        Evaluates a batch of states & actions:
          - log p(a|s)
          - reward value function
          - cost value function
          - distribution entropy
        """
        action_probs = self.actor(states)
        dist = Categorical(action_probs)

        logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        value_r = self.critic_r(states).squeeze(-1)  # shape (batch,)
        value_c = self.critic_c(states).squeeze(-1)  # shape (batch,)

        return logprobs, value_r, value_c, dist_entropy


#  SAFE PPO AGENT
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        K_epochs=10,
        cost_limit=0.1,
        alpha_lr=1e-2,
    ):
        """
        Args:
          state_dim:  dimension of observation vector
          action_dim: number of discrete actions
          lr_actor, lr_critic:  learning rates
          gamma: discount factor for reward/cost
          lam:   GAE lambda
          eps_clip: PPO clip parameter
          K_epochs: number of optimization epochs per update
          cost_limit: desired upper bound on average cost
          alpha_lr:   step size for Lagrange multiplier
        """
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = SafeRolloutBuffer()

        self.policy = SafeActorCritic(state_dim, action_dim)
        self.policy_old = SafeActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Combine parameters from all networks for one optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_r.parameters(), "lr": lr_critic},
                {"params": self.policy.critic_c.parameters(), "lr": lr_critic},
            ]
        )

        self.mse_loss = nn.MSELoss()

        # Safe RL additions
        self.alpha = 0.0  # Lagrange multiplier
        self.alpha_lr = alpha_lr  # step size to update alpha
        self.cost_limit = cost_limit

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy_old.to(self.device)
        self.name = "PPO_SAFE"

    def select_action(self, state):
        """Selects an action given state, using old policy for sampling."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            action, logprob, value_r, value_c = self.policy_old.act(state_t)

        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.value_r.append(value_r)
        self.buffer.value_c.append(value_c)

        return action.item()

    def update(self):
        """
        Run one PPO update (with K_epochs) on the entire buffer.
        Includes Lagrange-based penalty for costs.
        """
        # Convert lists to tensors
        old_states = torch.stack(self.buffer.states, dim=0).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device).detach()
        old_values_r = (
            torch.stack(self.buffer.value_r, dim=0).to(self.device).detach().squeeze(-1)
        )
        old_values_c = (
            torch.stack(self.buffer.value_c, dim=0).to(self.device).detach().squeeze(-1)
        )

        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        costs = np.array(self.buffer.costs, dtype=np.float32)
        dones = np.array(self.buffer.is_terminals, dtype=np.bool_)

        # ----------------------------
        # Compute reward advantages (GAE)
        # ----------------------------
        advantages_r = []
        gae_r = 0.0
        next_value_r = 0.0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value_r = 0.0
                gae_r = 0.0
            delta_r = rewards[i] + self.gamma * next_value_r - old_values_r[i].item()
            gae_r = delta_r + self.gamma * self.lam * gae_r
            advantages_r.insert(0, gae_r)
            next_value_r = old_values_r[i].item()
        advantages_r = torch.tensor(
            advantages_r, dtype=torch.float32, device=self.device
        )
        returns_r = advantages_r + old_values_r

        # Compute cost advantages (GAE)
        advantages_c = []
        gae_c = 0.0
        next_value_c = 0.0
        for i in reversed(range(len(costs))):
            if dones[i]:
                next_value_c = 0.0
                gae_c = 0.0
            delta_c = costs[i] + self.gamma * next_value_c - old_values_c[i].item()
            gae_c = delta_c + self.gamma * self.lam * gae_c
            advantages_c.insert(0, gae_c)
            next_value_c = old_values_c[i].item()
        advantages_c = torch.tensor(
            advantages_c, dtype=torch.float32, device=self.device
        )
        returns_c = advantages_c + old_values_c

        # Normalize the reward advantage if desired
        advantages_r = (advantages_r - advantages_r.mean()) / (
            advantages_r.std() + 1e-8
        )

        advantages_c = (advantages_c - advantages_c.mean()) / (
            advantages_c.std() + 1e-8
        )

        #  PPO TRAINING
        for _ in range(self.K_epochs):
            logprobs, v_r, v_c, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # ratio for clipping
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # surrogates
            surr1 = ratios * (advantages_r - self.alpha * advantages_c)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * (
                advantages_r - self.alpha * advantages_c
            )

            # final objective
            actor_loss = -torch.min(surr1, surr2).mean()  # note the negative sign

            # critic (reward)
            critic_loss_r = self.mse_loss(v_r, returns_r)
            # critic (cost)
            critic_loss_c = self.mse_loss(v_c, returns_c)

            loss = (
                actor_loss
                + 0.5 * (critic_loss_r + critic_loss_c)
                - 0.01 * dist_entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping if desired
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Update Lagrange multiplier alpha
        # measure average cost in the batch
        avg_cost = float(costs.mean())
        # simple primal-dual ascent on alpha
        self.alpha += self.alpha_lr * (avg_cost - self.cost_limit)
        # alpha should never be negative
        self.alpha = max(self.alpha, 0.0)

        # Clear buffer
        self.buffer.clear()

        return avg_cost  # might help logging, e.g. how we track cost per update

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_old.load_state_dict(torch.load(path, map_location=self.device))


###################################################
#  TRAINING LOOP EXAMPLE (PSEUDOCODE)
###################################################
def train_safe_ppo(env, max_episodes=2000, max_timesteps=1500):
    # hyperparams
    state_dim = env.state_dims
    action_dim = env.action_dims
    agent = SafePPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        K_epochs=10,
        cost_limit=0.1,
        alpha_lr=0.01,
    )

    for ep in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_cost = 0

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # gather transition
            agent.buffer.rewards.append(reward)
            cost = (
                info["cost"] if "cost" in info else (1.0 if env.already_crash else 0.0)
            )
            agent.buffer.costs.append(cost)
            agent.buffer.is_terminals.append(done)

            state = next_state
            episode_reward += reward
            episode_cost += cost

            if done:
                break

        # Update once per episode
        avg_cost = agent.update()

        print(
            f"Episode {ep}, reward={episode_reward:.2f}, ep_cost={episode_cost}, "
            f"avg_cost_in_update={avg_cost:.3f}, alpha={agent.alpha:.3f}"
        )

    # You can save the final policy
    agent.save("safe_ppo_final.pth")
    return agent
