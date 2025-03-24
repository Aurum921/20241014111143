from magent2.environments import battle_v4
import os
import cv2
from torch_model import QNetwork
import torch

if __name__ == "__main__":
    env = battle_v4.env(map_size=60, render_mode="human")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35
    
    # random policies
    frames = []
    env.reset()
    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("blue.pt", weights_only=True, map_location="cpu")
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

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
            else:
                action = env.action_space(agent).sample()

        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"random.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    print("Done random agents")
 
    # pretrained policies
    frames = []
    env.reset()
    from torch_model import QNetwork
    import torch

    q_network_blue_1 = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network_blue_1.load_state_dict(
        torch.load("blue.pt", weights_only=True, map_location="cpu")
    )

    q_network_red_1 = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network_red_1.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = q_network_blue_1(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                # action = env.action_space(agent).sample()
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values_red_1 = q_network_red_1(observation)
                action = torch.argmax(q_values_red_1, dim=1).numpy()[0]

        env.step(action)

        if agent == "blue_80":
            frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"pretrained.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    print("Done recording pretrained agents")

    env.close()
    


    from final_torch_model import QNetwork as FinalQNetwork
    # final policy
    frames = []
    env.reset()
    q_network_blue_2 = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network_blue_2.load_state_dict(
        torch.load("blue.pt", weights_only=True, map_location="cpu")
    )

    # q_network_red_2 = QNetwork(
    #     env.observation_space("red_0").shape, env.action_space("red_0").n
    # )
    # q_network_red_2.load_state_dict(
    #     torch.load("blue.pt", weights_only=True, map_location="cpu")
    # )

    q_network_red_2 = FinalQNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network_red_2.load_state_dict(
        torch.load("red_final.pt", weights_only=True, map_location="cpu")
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = q_network_blue_2(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                # action = env.action_space(agent).sample()
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values_red_2 = q_network_red_2(observation)
                action = torch.argmax(q_values_red_2, dim=1).numpy()[0]

        env.step(action)

        if agent == "blue_80":
            frames.append(env.render())

    # height, width, _ = frames[0].shape
    # out = cv2.VideoWriter(
    #     os.path.join(vid_dir, f"final_pretrained.mp4"),
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (width, height),
    # )
    # for frame in frames:
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame_bgr)
    # out.release()
    print("Done recording final_pretrained agents")

    env.close()


