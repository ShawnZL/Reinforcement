import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import utils

class PolicyNet(torch.nn.Module):
    '''定义策略网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # softmax 常用激活函数，可以将输入的原始数据转换为概率分布的形式
        return F.softmax(self.fc2(x), dim = 1) # 在第二个维度进行归一化处理，
        # 将左右两个方向选择改为归一化，例如[0.4571, 0.5429] [0.4366, 0.5634]

class ValueNet(torch.nn.Module):
    '''价值网络，输入状态，输出状态的价值'''
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype = torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states) # 时序差分误差
        # gather 函数的第一个参数是一个张量，表示待选取的值，第二个参数是一个索引张量，表示待选取的索引。在这里，actions 是一个张量，包含了要选择的动作的索引。
        # 根据代码中的 gather(1, actions)，第二个参数 actions 是一个大小为 [batch_size, 1] 的张量，包含了要选择的动作的索引。
        # 而第一个参数的维度 1 表示要在第 1 维上进行索引操作，也就是在每个样本的概率分布中选择对应的动作概率。
        # [batch_size, num_actions] [批量大小,动作的数量]
        # print(actions) 这里是记录我们所选择的动作，然后使用gather进行一种索引操作
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # print(log_probs)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad() # 进行梯度清零
        self.critic_optimizer.zero_grad() # 进行梯度清零
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name)
obs, info = env.reset(seed = 0)
state_dim = obs.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

return_list = utils.train_on_policy_agent(env, agent, num_episodes)