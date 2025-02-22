import cv2
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import MoveClassifier
from utils import display_observation, image_preprocess, policy_forward, discount_rewards

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
display = True
# hyperparameters
hidden_dim = 256
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
weight_decay = 1e-2

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
model = MoveClassifier(num_features=D, hidden_size=hidden_dim)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.BCELoss()


env = gym.make("Pong-v4", render_mode="rgb_array")
observation = env.reset(seed=42)
observation = observation[0]  # first observation is tuple of [numpy image, game info]



prev_x = None  # used in computing the difference frame
input_data, gt_labels, action_rewards = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    # preprocess the observation, set input to network to be difference image
    cur_x = image_preprocess(observation, device=device)
    input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D).to(device)
    prev_x = cur_x

    action = policy_forward(preprocessed_observation=input_x, model=model)  # Sample a random action

    # record value
    y = 1 if action == 2 else 0  # a "fake label"
    input_data.append(input_x)  # observation
    gt_labels.append(y)  # label

    # step the environment and get new measurements
    observation, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward

    action_rewards.append(reward)  # record reward after previous action

    if display:  # hold any key pressed to watch the game
        display_observation(observation=observation)

    if terminated or truncated:  # an episode finished
        episode_number += 1


        # stack inputs, targets and rewards into a batch
        inputs = torch.stack([torch.tensor(i) for i in input_data]).to(device)
        discounted_r = discount_rewards(rewards=gt_labels, gamma=gamma, device=device)
        targets = torch.stack([torch.tensor(i) for i in gt_labels]).to(device)



        # calculate loss and grads

        # backward only after batch size number of episodes




        print("Game terminated with reward", reward)
        break

print("Game Over")
env.close()
cv2.destroyAllWindows()
# plt.show()
