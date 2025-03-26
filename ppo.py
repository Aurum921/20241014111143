import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, observation_shape, n_hidden, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3, padding=1),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 84),
            nn.LayerNorm(84),
            nn.Tanh(),
        )
        self.last_layer = nn.Linear(84, action_shape)

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        self.last_latent = x
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x + self.last_latent
        x = x.reshape(batchsize, -1)
        x = self.network(x)
        x = self.last_layer(x)
        x = F.softmax(x, dim=1)
        return x


# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, observation_shape, n_hidden):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 84),
            nn.LayerNorm(84),
            nn.Tanh(),
        )
        self.last_layer = nn.Linear(84, 1)

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        x = self.network(x)
        # self.last_latent = x
        return self.last_layer(x)


# ----------------------------------- #
# 构建模型
# ----------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # self.actor.load_state_dict(torch.load("actor8.pt"))
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # self.critic.load_state_dict(torch.load("critic8.pt"))
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        """state1 = np.array(list(state))
        state2 = torch.from_numpy(state1)
        state3 = torch.unsqueeze(state2, 0)"""
        # state = torch.tensor(state[len(state), :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action

    # 训练
    def learn(self, transition_dict, train_critic=False):
        # 提取数据集
        # states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        states = torch.stack(transition_dict['states'], dim=0).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        next_states = torch.stack(transition_dict['next_states'], dim=0).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式 :计算优势函数估计，使用时间差分误差和上一个时间步的优势函数估计
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if train_critic:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            return self.actor, self.critic


def find_duplicates(numbers):
    seen = set()
    duplicates = set()

    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)

    return list(duplicates)