import argparse

import ale_py
import gymnasium as gym
import numpy as np
import torch

from utils import display_observation, image_preprocess


@torch.no_grad()
def main(args):
    # init the parameters
    display = True
    D = 80 * 80  # input dimensionality: 80x80 grid
    prev_x = None
    reward_sum = 0

    # load model
    model = torch.load(args.model_path, weights_only=False)
    model.eval()
    model.to(args.device)

    # init the game
    env = gym.make("Pong-v4", render_mode="rgb_array")  # render_mode="human" option fails on my PC, thus used opencv
    observation = env.reset(seed=42)
    observation = observation[0]  # first observation is tuple of [numpy image, game info]

    while True:
        # preprocess the observation, set input to network to be difference image
        cur_x = image_preprocess(observation, device=args.device)
        input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D).to(args.device)
        prev_x = cur_x

        # model forward pass
        output = model(input_x)
        action = 2 if np.random.uniform() < output.item() else 3  # roll the dice!

        # step the environment and get new measurements
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        # display game if needed
        if display:
            display_observation(observation=observation)

        if terminated or truncated:  # an episode finished, someone reached 22 scores
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
