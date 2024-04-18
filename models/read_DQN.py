from .DQN_Atari import DQNAgent
from gymnasium import spaces
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

class Env:
    def __init__(self):
        self.seed = 525
        self.seed = 525
        
        channel_n = 1
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, 
                                            high=255,
                                            shape=(channel_n, 84, 84), 
                                            dtype=np.uint8)
        
def set_up_agent(model_dir = "/home/neaf2080/code/yen/webots_arm/integration/controllers/train_RL_env/checkpoint/Apr07_H16_M46_S15_cube_cam_neaf-3090_001/good_model_state_dict.pt"):
    env = Env()

    agent = DQNAgent(env=env)     
    agent.DQN.load_state_dict(torch.load(model_dir))
    
    agent.DQN.requires_grad_(False)
    
    # agent.DQN.eval()
    
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
    

