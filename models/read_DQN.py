
from gymnasium import spaces
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import os, glob
from tqdm import tqdm

class Env:
    def __init__(self, image_size=84):
        self.seed = 525
        self.seed = 525
        
        channel_n = 1
        if image_size == 256:
            action_n = 78
        elif image_size == 84:
            action_n = 6
        self.action_space = spaces.Discrete(action_n)
        self.observation_space = spaces.Box(low=0, 
                                            high=255,
                                            shape=(image_size, image_size, channel_n), 
                                            dtype=np.uint8)
        
def set_up_agent(image_size=84, which_DQN="007", gpu_ids=[]):
    
    if image_size == 84:
        from .DQN_Atari import DQNAgent
        model_dir = "./RL_model/DQN_gray/Apr15_H17_M58_S28_cube_gray_neaf2080_002/good_model_state_dict.pt"
    elif image_size == 256:
        from .DQN_Atari_256 import DQNAgent
        model_dir = glob.glob(f"./RL_model/DQN_gray_256/*_{which_DQN}/good_model_state_dict.pt")[0]
        print(model_dir)
    else:
        print("image_size must be 84 or 256")
        exit()
    
    # ----------------------------
    # env 
    env = Env(image_size)
    # ----------------------------
    # agent
    agent = DQNAgent(env=env)       
    if len(gpu_ids) == 0:
        agent.DQN.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))    
        agent.DQN.cpu() 
    else:
        agent.DQN.load_state_dict(torch.load(model_dir))    
    agent.DQN.requires_grad_(False)
    # agent.DQN.eval()
    # ----------------------------
    
    return agent

def deal_with_image(state_dir: str = "/home/neaf2080/code/yen/pytorch-CycleGAN-and-pix2pix/Robotic_arm_image_84/Simulation/train/107.png"):
    state_image = cv2.imread(state_dir)
    state_image = cv2.cvtColor(state_image, cv2.COLOR_BGR2GRAY) 

    state_image = np.expand_dims(state_image, axis=-1) 
    return state_image

def get_q_values(agent, state_image):
    # state_image = deal_with_image(state_dir)

    state_tensor = agent.observe(state_image)
    q_values = agent.value(state_tensor).cpu().detach().numpy()
    return q_values

def save_image_array_pair(state_image, q_values, o_image_dir):
    # 假設你有一張影像和對應的 NumPy array 數據
    image_data = state_image.copy()
    array_data = q_values.copy()

    # 創建一個結構化數據
    structured_data = {'image': image_data, 'array': array_data}

    # 將結構化數據保存到文件中
    np.savez(o_image_dir, **structured_data)
    

