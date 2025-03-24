from magent2.environments import battle_v4
from org_ppo import PPO
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_model import QNetwork


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    map_size = 16
    step_reward = -0.1
    attack_penalty = -0.1
    attack_opponent_reward = 3.0
    dead_penalty = -3.0
    max_cycles = 300
    # render_mode = "human"
    render_mode = "rgb_array"
    trapped_episodes = 0

    env = battle_v4.env(map_size=map_size,
                        render_mode=render_mode,
                        step_reward=step_reward,
                        dead_penalty=dead_penalty,
                        attack_penalty=attack_penalty,
                        attack_opponent_reward=attack_opponent_reward,
                        max_cycles=max_cycles)

    # ----------------------------------------- #
    # 参数设置
    # ----------------------------------------- #

    num_episodes = 1000  # 总迭代次数
    gamma = 0.99  # 折扣因子
    actor_lr = 1e-5  # 策略网络的学习率
    critic_lr = 1e-5  # 价值网络的学习率
    n_hiddens = 256  # 隐含层神经元个数
    return_list = []  # 保存每个回合的return

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("blue.pt", weights_only=True, map_location="cpu")
    )

    # ----------------------------------------- #
    # 环境加载
    # ----------------------------------------- #

    n_states_shape = env.observation_space("red_0").shape
    n_actions = env.action_space("red_0").n

    # ----------------------------------------- #
    # 模型构建
    # ----------------------------------------- #

    policy = PPO(n_states=n_states_shape,  # 状态数
                 n_hiddens=n_hiddens,  # 隐含层数
                 n_actions=n_actions,  # 动作数
                 actor_lr=actor_lr,  # 策略网络学习率
                 critic_lr=critic_lr,  # 价值网络学习率
                 lmbda=0.98,  # 优势函数的缩放因子
                 epochs=2,  # 一组序列训练的轮次
                 eps=0.1,  # PPO中截断范围的参数
                 gamma=gamma,  # 折扣因子
                 device=device)

    # ----------------------------------------- #
    # 训练--回合更新 on_policy
    # ----------------------------------------- #

    for i in range(num_episodes):

        reward_dict = {}

        # full_dict = {}
        full_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        env.reset()

        for agent in env.agent_iter():

            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                action = None  # this agent has died
            else:
                agent_handle = agent.split("_")[0]
                if agent_handle == "blue":
                    observation = (
                        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )
                    with torch.no_grad():
                        q_values = q_network(observation)
                    action = torch.argmax(q_values, dim=1).numpy()[0]
                    # action = env.action_space(agent).sample()
                else:
                    observation = (
                        torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )

                    # observation = observation.reshape(-1)
                    action = policy.take_action(observation[0])

            env.step(action)

            if agent_handle == "red":

                if agent not in reward_dict:
                    reward_dict[agent] = 0

                # # 构造数据集，保存每个回合的状态数据
                # if agent not in full_dict:
                #     full_dict[agent] = {
                #         'states': [],
                #         'actions': [],
                #         'next_states': [],
                #         'rewards': [],
                #         'dones': [],
                #     }
                #
                # if not termination:
                #     next_observation, _, _, _, _ = env.last()
                #     next_observation = next_observation.reshape(-1)
                #     full_dict[agent]['states'].append(observation)
                #     full_dict[agent]['actions'].append(action)
                #     full_dict[agent]['next_states'].append(next_observation)
                #     full_dict[agent]['rewards'].append(reward)
                #     full_dict[agent]['dones'].append(termination)

                # 构造数据集，保存每个回合的状态数据

                if not termination:
                    next_observation, next_reward, next_termination, _, _ = env.last()
                    next_observation = (
                        torch.Tensor(next_observation).float().permute([2, 0, 1]).unsqueeze(0)
                    )

                    full_dict['states'].append(observation[0])
                    full_dict['actions'].append(action)
                    full_dict['next_states'].append(next_observation[0])
                    full_dict['rewards'].append(reward)
                    full_dict['dones'].append(termination)

                    # 累计回合奖励
                    reward_dict[agent] += reward

                if termination:
                    full_dict['states'].append(next_observation[0])
                    full_dict['actions'].append(None)
                    full_dict['next_states'].append(next_observation[0])
                    full_dict['rewards'].append(next_reward)
                    full_dict['dones'].append(termination)

                    reward_dict[agent] += next_reward

        actor, critic = policy.learn(full_dict)

        # 保存每个回合的return
        # return_list.append(reward_dict[agent])
        # 模型训练
        # 打印回合信息
        print(f'iter:{i}, return:{np.array(list(reward_dict.values())).mean()}')
        return_list.append(np.array(list(reward_dict.values())).mean())
        if abs(np.array(return_list[-100:]).mean() - return_list[-1]) < 1.0 and len(return_list) > 300:
            trapped_episodes += 1

        # if trapped_episodes > 50:
        #     print("early stop.")
        #     break

    # -------------------------------------- #
    # 绘图
    # -------------------------------------- #

    torch.save(actor.state_dict(), "actor.pt")
    torch.save(critic.state_dict(), "critic.pt")
    plt.plot(return_list)
    print(return_list)
    plt.title('return')
    plt.show()
