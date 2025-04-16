import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import utils

device = utils.get_device()

################################## Safe PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.cost_values = []  # Cost values for safety constraints
        self.costs = []  # Costs received from environment
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.cost_values[:]
        del self.costs[:]


class SafeActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(SafeActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        
        # Value Critic network (for rewards)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Cost Critic network (for safety constraints)
        self.cost_critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling SafeActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

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
        cost_val = self.cost_critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach(), cost_val.detach()
    
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
        cost_values = self.cost_critic(state)
        
        return action_logprobs, state_values, cost_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, cost_gamma, K_epochs, 
                 eps_clip, has_continuous_action_space, cost_limit=25.0, lagrangian_multiplier_init=0.01, 
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.cost_gamma = cost_gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Safety constraint parameters
        self.cost_limit = cost_limit  # Maximum allowed cost
        self.lagrangian_multiplier = lagrangian_multiplier_init  # Lagrangian multiplier
        self.lambda_lr = 0.99  # Learning rate for Lagrangian multiplier
        
        self.buffer = RolloutBuffer()

        self.policy = SafeActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.cost_critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = SafeActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.name = "PPO_SAFE_2"
            
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling SafePPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling SafePPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, cost_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            self.buffer.cost_values.append(cost_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val, cost_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            self.buffer.cost_values.append(cost_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Monte Carlo estimate of costs
        costs = []
        discounted_cost = 0
        for cost, is_terminal in zip(reversed(self.buffer.costs), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_cost = 0
            discounted_cost = cost + (self.cost_gamma * discounted_cost)
            costs.insert(0, discounted_cost)
        
        # Normalizing the rewards and costs
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        costs = torch.tensor(costs, dtype=torch.float32).to(device)
        costs = (costs - costs.mean()) / (costs.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_cost_values = torch.squeeze(torch.stack(self.buffer.cost_values, dim=0)).detach().to(device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        
        # Calculate cost advantages
        cost_advantages = costs.detach() - old_cost_values.detach()
        
        # Calculate mean cost
        mean_cost = costs.mean().item()
        
        # Use raw cost sum instead of normalized costs for constraint violation
        raw_cost_mean = sum(self.buffer.costs) / len(self.buffer.costs)
        cost_violation = raw_cost_mean - self.cost_limit

        # Update with larger learning rate and ensure it responds to violation
        self.lambda_lr = 0.5  # Increase from 0.05
        self.lagrangian_multiplier = max(0.0, self.lagrangian_multiplier + self.lambda_lr * cost_violation)
        print(f"Cost violation: {cost_violation:.2f}, Updated lambda: {self.lagrangian_multiplier:.6f}")
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, cost_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            cost_values = torch.squeeze(cost_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss for rewards
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Finding Surrogate Loss for costs
            cost_surr = ratios * cost_advantages
            
            # Final loss of constrained objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) \
                   + 0.5 * self.MseLoss(cost_values, costs) \
                   + self.lagrangian_multiplier * cost_surr \
                   - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()
        
        return mean_cost, self.lagrangian_multiplier
            
    def save(self, checkpoint_path):
        # Save model state with the Lagrangian multiplier
        torch.save({
            'model_state_dict': self.policy_old.state_dict(),
            'lagrangian_multiplier': self.lagrangian_multiplier,
            'cost_limit': self.cost_limit
        }, checkpoint_path)
   
    def load(self, checkpoint_path):
        # Load the saved state
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # Load model state
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Load Lagrangian multiplier and cost limit if available
        if 'lagrangian_multiplier' in checkpoint:
            self.lagrangian_multiplier = checkpoint['lagrangian_multiplier']
        
        if 'cost_limit' in checkpoint:
            self.cost_limit = checkpoint['cost_limit']