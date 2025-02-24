
import argparse
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import MoveClassifier
from utils import display_observation, image_preprocess, policy_forward, discount_rewards, init_all_seeds
import torch

@torch.no_grad()
def main(args):
    model = torch.load(args.model_path, weights_only=False)
    model.eval()
    model.to(args.device)

    env = gym.make("Pong-v4", render_mode="rgb_array")  # render_mode="human" option fails on my PC, thus used opencv
    observation = env.reset()
    observation = observation[0]  # first observation is tuple of [numpy image, game info]
    display = True
    D = 80 * 80  # input dimensionality: 80x80 grid
    prev_x = None
    reward_sum = 0

    while True:
        # preprocess the observation, set input to network to be difference image
        cur_x = image_preprocess(observation, device=args.device)
        input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D).to(args.device)
        prev_x = cur_x

        action = policy_forward(preprocessed_observation=input_x, model=model)  # Sample a random action

        # step the environment and get new measurements
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward

        if display:  # hold any key pressed to watch the game
            display_observation(observation=observation)

        if terminated or truncated:  # an episode finished, 22 played games
            print('Episode total reward:', reward_sum)
            break



def parse_args():
    ap = argparse.ArgumentParser('Evaluate Parser')
    ap.add_argument('--model_path', type=str, default='best_reward_model.pth',
                    help="Path to the model .pth file")
    ap.add_argument('--device', type=str, default='cuda:0',
                    help="Device to use")
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)