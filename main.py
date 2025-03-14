import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import MoveClassifier
from utils import display_observation, image_preprocess, policy_forward, discount_rewards, init_all_seeds, plot_rewards

display = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
init_all_seeds(42)
# hyperparameters
hidden_dim = 200
batch_size = 64  # how many episodes to do a param update after
learning_rate = 0.01
gamma = 0.99  # discount factor for reward
weight_decay = 1e-4

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
model = MoveClassifier(num_features=D, hidden_size=hidden_dim)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.BCELoss(reduction='none')

env = gym.make("Pong-v4", render_mode="rgb_array")  # render_mode="human" option fails on my PC, thus used opencv
observation = env.reset(seed=42)
observation = observation[0]  # first observation is tuple of [numpy image, game info]

prev_x = None  # used in computing the difference frame
input_data, gt_labels, action_rewards = [], [], []
reward_sum = 0
batch_mean_reward = []
exp_mean_reward = None
best_batch_mean_reward = float('-inf')
batch_loss = []
batch_weighted_loss = []
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

    if terminated or truncated:  # an episode finished, multiple played games
        episode_number += 1
        batch_mean_reward.append(reward_sum)

        # stack inputs, targets and rewards into a batch
        inputs = torch.stack([torch.tensor(i) for i in input_data]).to(device)
        discounted_r = discount_rewards(rewards=action_rewards, gamma=gamma, device=device)
        targets = torch.stack([torch.tensor(i) for i in gt_labels]).to(device).float()

        # Calculate Loss and Grads
        model_preds = model(inputs)
        loss = loss_fn(model_preds.squeeze(-1), targets) / batch_size  # batch_size == grad accumulation iterations
        weighted_loss = loss * discounted_r
        weighted_loss = weighted_loss.sum()  # not average, since each episode is used as a one data sample
        weighted_loss.backward()

        # track losses
        batch_loss.append(loss.mean().item())  # reporting mean loss instead of the sum
        batch_weighted_loss.append(weighted_loss.item())
        exp_mean_reward = reward_sum if exp_mean_reward is None else exp_mean_reward * 0.99 + reward_sum * 0.01

        # update after batch size of episodes
        if episode_number % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_mean_reward = np.mean(batch_mean_reward)
            plot_rewards(episode=episode_number, exp_reward=exp_mean_reward)
            print(
                f"Episode {episode_number}:"
                f"\n\t\t Batch mean reward: {batch_mean_reward:.4f} "
                f":: Exp mean reward: {exp_mean_reward:.7f} "
                f":: Loss: {np.mean(batch_loss):.7f} "
                f":: Weighted loss: {np.mean(batch_weighted_loss):.7f} ")

            if batch_mean_reward > best_batch_mean_reward:
                best_batch_mean_reward = batch_mean_reward
                torch.save(model, f"best_reward_model.pth")

            batch_mean_reward = []
            batch_loss = []
            batch_weighted_loss = []

        # reset the values
        input_data, gt_labels, action_rewards = [], [], []
        observation = env.reset()  # reset env
        observation = observation[0]  # first observation is tuple of [numpy image, game info]
        prev_x = None
        reward_sum = 0

print("Game Over")
env.close()
cv2.destroyAllWindows()
