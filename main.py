import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import MoveClassifier
from utils import display_observation, image_preprocess, policy_forward, discount_rewards

display = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# hyperparameters
hidden_dim = 10
batch_size = 32  # how many episodes to do a param update after
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
weight_decay = 1e-4

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
model = MoveClassifier(num_features=D, hidden_size=hidden_dim)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.BCELoss(reduction='none')

env = gym.make("Pong-v4", render_mode="rgb_array")
observation = env.reset(seed=42)
observation = observation[0]  # first observation is tuple of [numpy image, game info]

prev_x = None  # used in computing the difference frame
input_data, gt_labels, action_rewards = [], [], []
best_mean_episode_reward = float('-inf')
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

    action_rewards.append(reward)  # record reward after each action

    if display:  # hold any key pressed to watch the game
        display_observation(observation=observation)

    if terminated or truncated:  # an episode finished, finishes 22 played games
        episode_number += 1
        episode_mean_reward = np.mean(action_rewards)

        # stack inputs, targets and rewards into a batch
        inputs = torch.stack([torch.tensor(i) for i in input_data]).to(device)
        discounted_r = discount_rewards(rewards=action_rewards, gamma=gamma, device=device)
        targets = torch.stack([torch.tensor(i) for i in gt_labels]).to(device).float()

        # calculate loss and grads
        model_preds = model(inputs)
        # -1 multiplication is used to go uphill, since we want maximization of positive reward/action probability
        loss = -1 * loss_fn(model_preds.squeeze(-1), targets) * discounted_r
        loss = loss.mean()
        loss.backward()

        # backward only after batch size number of episodes
        if episode_number % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(
                f"Episode number {episode_number}: \n\t\tMean episode reward: {episode_mean_reward:.4f} :: loss: {loss.item():.4f}")

        if episode_mean_reward > best_mean_episode_reward:
            best_mean_episode_reward = episode_mean_reward
            torch.save(model, f"best_mean_episode_model.pth")

        # reset the values
        input_data, gt_labels, action_rewards = [], [], []
        observation = env.reset()  # reset env
        observation = observation[0]
        prev_x = None

print("Game Over")
env.close()
cv2.destroyAllWindows()
