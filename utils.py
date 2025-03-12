import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def display_observation(observation: np.ndarray) -> None:
    rgb_img = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    rgb_img = cv2.resize(rgb_img, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Pong", rgb_img)
    cv2.waitKey(1)
    time.sleep(0.025)  # Slow down the rendering


def image_preprocess(observation: np.ndarray, device):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    obs = observation[35:195]  # crop scores table from img
    obs = obs[::2, ::2, 0]  # downsample by factor of 2
    obs[obs == 144] = 0  # erase background (background type 1)
    obs[obs == 109] = 0  # erase background (background type 2)
    obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1
    obs = obs.astype(np.float32).ravel()
    output_tensor = torch.from_numpy(obs).float().to(device)
    return output_tensor


def discount_rewards(rewards: list, gamma: float, device) -> torch.Tensor:
    """
    Takes 1D float array of rewards and compute discounted reward

    Args:
        rewards:
        gamma:
        device:

    Returns:
        Tensor of discounted rewards
    """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):  # reversed loop, from the last item index to the first
        if rewards[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    discounted_r = torch.tensor(discounted_r).to(device)
    # normalize discounted rewards for stable updates
    discounted_r -= torch.mean(discounted_r)
    discounted_r /= torch.std(discounted_r)
    return discounted_r


@torch.no_grad()
def policy_forward(preprocessed_observation: torch.Tensor, model: nn.Module) -> int:
    """
    move up: 2
    move down: 3
    Args:
        preprocessed_observation:  preprocessed current observation - preprocessed previous observation
        model:  policy model

    Returns:
        action index
    """
    output = model(preprocessed_observation)
    action = 2 if np.random.uniform() < output.item() else 3  # roll the dice!
    return action


def init_all_seeds(seed_value):
    """
    Initialize all seeds for reproducibility in Python functions.
    Args:
        seed_value (int): The seed value to initialize the random number generators.
    """
    # For Python's random module
    random.seed(seed_value)
    # For numpy
    np.random.seed(seed_value)
    # For PyTorch (if using)
    if torch.cuda.is_available():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    else:
        torch.manual_seed(seed_value)


exp_rewards = []
episodes = []


def plot_rewards(episode: int, exp_reward: float) -> None:
    episodes.append(episode)
    exp_rewards.append(exp_reward)

    plt.figure(figsize=(13, 10))  # Set figure size
    plt.plot(episodes, exp_rewards, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Exponentially weighted reward')
    # plt.legend()
    # Save the plot as an image file
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')
    plt.close()
