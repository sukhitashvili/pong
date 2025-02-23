import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from model import MoveClassifier
from utils import display_observation, image_preprocess, policy_forward, discount_rewards, init_all_seeds

display = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
init_all_seeds(42)
# hyperparameters
hidden_dim = 200
batch_size = 10  # how many episodes to do a param update after
learning_rate = 0.001
gamma = 0.99  # discount factor for reward
weight_decay = 1e-4

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
model = MoveClassifier(num_features=D, hidden_size=hidden_dim)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # , weight_decay=weight_decay)
loss_fn = nn.BCELoss(reduction='none')

env = gym.make("Pong-v4", render_mode="rgb_array")
observation = env.reset(seed=42)
observation = observation[0]  # first observation is tuple of [numpy image, game info]

prev_x = None  # used in computing the difference frame
input_data, gt_labels, action_rewards = [], [], []
reward_sum = 0
batch_reward_sum = 0
best_batch_reward = float('-inf')
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
        batch_reward_sum += reward_sum

        # stack inputs, targets and rewards into a batch
        inputs = torch.stack([torch.tensor(i) for i in input_data]).to(device)
        discounted_r = discount_rewards(rewards=action_rewards, gamma=gamma, device=device)
        targets = torch.stack([torch.tensor(i) for i in gt_labels]).to(device).float()

        # Calculate Loss and Grads
        model_preds = model(inputs)
        # -1 multiplication is used to go uphill, since we want maximization of positive reward/action probability
        loss = -1 * loss_fn(model_preds.squeeze(-1), targets) / batch_size  # batch_size == grad accumulation iterations
        weighted_loss = loss * discounted_r
        weighted_loss = weighted_loss.mean()
        weighted_loss.backward()

        # update after batch size of episodes
        if episode_number % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(
                f"Episode number {episode_number}:"
                f"\n\t\t Batch reward sum: {batch_reward_sum:.4f} "
                f":: loss: {loss.mean().item():.7f} "
                f":: weighted loss: {weighted_loss.item():.7f}")

            if batch_reward_sum > best_batch_reward:
                best_batch_reward = batch_reward_sum
                torch.save(model, f"best_reward_model.pth")

            batch_reward_sum = 0

        # reset the values
        input_data, gt_labels, action_rewards = [], [], []
        observation = env.reset()  # reset env
        observation = observation[0]  # first observation is tuple of [numpy image, game info]
        prev_x = None
        reward_sum = 0

print("Game Over")
env.close()
cv2.destroyAllWindows()
