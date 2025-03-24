from magent2.environments import battle_v4
import os
from torch_model import QNetwork
import torch
import numpy as np
from ppo import PolicyNet
import copy

if __name__ == "__main__":

    env = battle_v4.env(map_size=16, render_mode="human", step_reward=-0.005,
                        dead_penalty=-0.3, attack_penalty=-0.1, attack_opponent_reward=0.5,
                        max_cycles=1000)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_dict = {}

    # random policies
    frames = []
    env.reset()
    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt")
    )

    ppo_network = PolicyNet(
        (13, 13, 6), 256, env.action_space("red_0").n
    )
    ppo_network.load_state_dict(
        torch.load("actor7.pt")
    )
    # ppo_network = torch.load("actor2.pt").to(device)
    # if torch.cuda.is_available():
    #     ppo_network = ppo_network.to('cuda')
    win = 0
    for i in range(500):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None  # this agent has died
            else:
                if agent not in reward_dict:
                    reward_dict[agent] = 0
                agent_handle = agent.split("_")[0]
                if agent_handle == "red":
                    observation = (
                        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )
                    noise = np.random.normal(loc=0, scale=0, size=(1, 1, 13, 13)).astype(np.float32)
                    noise = np.clip(noise, 0, 0.5)
                    noise = torch.tensor(noise)
                    observation = torch.cat((observation, noise), dim=1)

                    with torch.no_grad():
                        probs = ppo_network(observation[0])
                        # 创建以probs为标准的概率分布
                    action_list = torch.distributions.Categorical(probs)
                    # 依据其概率随机挑选一个动作
                    action = action_list.sample().item()
                    if not truncation:
                        reward_dict[agent] += reward
                    # action = env.action_space(agent).sample()
                else:
                    observation = (
                        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )
                    with torch.no_grad():
                        q_values = q_network(observation)
                    action = torch.argmax(q_values, dim=1).numpy()[0]
                    # action = env.action_space(agent).sample()

            env.step(action)
            if len(env.agents) > 1:
                winners = copy.deepcopy(env.agents)

        if termination:
            agent_handle = winners[0].split("_")[0]
            if agent_handle == "red":
                win += 1
                print('Round ' + str(i) + ' Red Win')
            if agent_handle == "blue":
                print('Round ' + str(i) + ' Blue Win')

    print(win)
