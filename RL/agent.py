import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using PPO instead of Deep-Q network bcoz:
#   - more stable - bcoz of the clipping function of the epsilon (0.2). Which means policy could not move beyond that factor of 0.8 to 1.2
#   -             - and the min function min(policy, clipped_value), which reduces the risk of moving further
#
#   - When the advantages > 0, which means the action is good, the update is minimal (stable)
#   - BUT when the advantages < 0, which means the action is bad, there is no limit on penalty
#   - By doing all this minimal update, what happen is we utilize our good sample that we saw once, 
#       and confirm it by doing it more time (data efficient compared to other on policy)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, nb_actions):
        super(ActorNetwork, self).__init__()

        self.linear1 = nn.Linear(in_size*nb_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, nb_actions)
        self.nb_actions = nb_actions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        state = torch.flatten(state)
        actions = self.relu(self.linear1(state))
        actions = self.tanh(self.linear2(actions))
        return actions


class CriticNetwork(nn.Module):
    def __init__(self, in_size, nb_actions):
        super(CriticNetwork, self).__init__()

        self.linear1 = nn.Linear(in_size*nb_actions, 1)
        self.tanh = nn.Tanh()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        state = torch.flatten(state)
        value = self.tanh(self.linear1(state))
        return value


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        actor_cfg = {
            'in_size':state_dim,
            'hidden_size':200,
            'nb_actions':action_dim
        }
        critic_cfg = {
            'in_size':state_dim,
            'nb_actions':action_dim
        }
        self.actor = ActorNetwork(**actor_cfg)
        self.critic = CriticNetwork(**critic_cfg)

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        # torch.distributions.Multivariatenormal is used to sample an action from the action probabilities computed by the actor network
        # The sample() method select actions based on given probablity. This is to introduce explore and exploit
        # covariance matrix is the one to be changed if we want to control exploration or exploitation more
        # lower variance means explore less
        # But cov_mat was set by self.action_var, which was set by action_std^2.
        # Therefore action_std play the biggest role in trading off exploration and exploitation
        # action_std comes from the terminal (args.action_std)
        # lower args.action_std, lower variance, explore less, exploit more
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, states, action):

        action_list = []
        for state in states:
            actions = self.actor(state)
            action_list.append(actions)

        action_mean = torch.stack(action_list)
        action_mean = torch.mean(action_mean, dim=0)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy



class Agent:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, seed):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        torch.manual_seed(seed)
    
    def select_action(self, state, memory):
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)   # Prioritize newer reward
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        #old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        #old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        #old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        """ print(f"torch.stack memory.states: {type(memory.states)}\n{torch.stack(memory.states)}")
        print(f"torch.stack memory.actions: {type(memory.actions)}\n{torch.stack(memory.actions)}")
        print(f"torch.stack memory.logprobs: {type(memory.logprobs)}\n{torch.stack(memory.logprobs)}") """



        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):

            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            """ print("\n")
            print(f"old_states:\t {old_states}")
            print(f"state_values:\t {state_values}")
            print(f"rewards:\t {rewards}")
            print(f"advantages:\t {advantages}")
            print(f"ratios:\t {ratios}")
            print(f"ratios * advantages:\t {surr1}")
            print(f"surr2 torch.clamp:\t {surr2}") """
            if _ % 1 == 0:
                print('Epoches {} \t loss: {} \t '.format(_, loss.mean()))

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())





            

    
