import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym


# 定义一个简单的神经网络作为策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, state):
        return torch.softmax(self.fc(state), dim=-1)


# 训练函数
def train(policy_net, optimizer, env, num_episodes=100):
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state)
            action_probs = policy_net(state)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    return episode_rewards


# 实验比较不同学习率的性能
def compare_learning_rates(env):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    num_episodes = 100

    # 不同学习率的设置
    learning_rates = [0.1, 1e-2, 0.005]
    rewards_by_lr = {}

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")

        # 创建策略网络和优化器
        policy_net = PolicyNetwork(input_size, output_size)
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)

        # 训练模型
        episode_rewards = train(policy_net, optimizer, env, num_episodes=num_episodes)
        rewards_by_lr[lr] = episode_rewards

    return rewards_by_lr


# 在 CartPole 环境中进行实验
env = gym.make('CartPole-v1')
rewards_by_lr = compare_learning_rates(env)

# 绘制图表
plt.figure(figsize=(10, 6))

for lr, rewards in rewards_by_lr.items():
    plt.plot(rewards, label=f"LR={lr}")

plt.title("Performance Comparison with Different Learning Rates")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.show()
