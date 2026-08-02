"""
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
    #model = torch.load(args.model_path, weights_only=False)
    ##model.eval()
    #model.to(args.device)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.load(
        args.model_path,
        map_location=device,
        weights_only=False
    )

    model.eval()
    model.to(device)

    # init the game
    env = gym.make("Pong-v4", render_mode="rgb_array")  # render_mode="human" option fails on my PC, thus used opencv
    observation = env.reset(seed=42)
    observation = observation[0]  # first observation is tuple of [numpy image, game info]

    while True:
        # preprocess the observation, set input to network to be difference image
        cur_x = image_preprocess(observation, device=device)
        input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D, device=device)
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
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help="Device to use")
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
"""

import argparse
import cv2
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model only if evaluating the AI
    if args.mode == "ai":
        model = torch.load(
            args.model_path,
            map_location=device,
            weights_only=False
        )
        model.eval()
        model.to(device)
    else:
        print("=====================================================")
        print("HUMAN MODE SELECTED")
        print("Control the right paddle using your KEYBOARD.")
        print(" - Press 'W' to move UP")
        print(" - Press 'S' to move DOWN")
        print(" - Press 'Q' to QUIT")
        print("Make sure the OpenCV game window is in focus!")

    # init the game
    env = gym.make("Pong-v4", render_mode="rgb_array")
    observation, info = env.reset(seed=42)

    # Initialize OpenCV window to capture keyboard input
    cv2.namedWindow("Pong")

    while True:
        if args.mode == "ai":
            # preprocess the observation, set input to network to be difference image
            cur_x = image_preprocess(observation, device=device)
            input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D, device=device)
            prev_x = cur_x

            # model forward pass
            output = model(input_x)
            action = 2 if np.random.uniform() < output.item() else 3  # roll the dice!
            
            # Allow user to quit early
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            # Take user input via KEYBOARD
            # A 30ms wait provides a playable ~33 FPS for the human
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('w'):
                action = 2  # UP
            elif key == ord('s'):
                action = 3  # DOWN
            elif key == ord('q'):
                break       # QUIT
            else:
                action = 0  # NOOP

        # step the environment and get new measurements
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        
        # display game if needed
        if display:
            # Convert RGB array from Gymnasium to BGR for OpenCV
            #bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            #cv2.imshow("Pong", bgr_image)

            bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

            # Scale by 4x
            display_image = cv2.resize(
            bgr_image,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow("Pong", display_image)

        if terminated or truncated:  # an episode finished, someone reached 21 scores
            print('Episode total reward:', reward_sum)
            break

    cv2.destroyAllWindows()
    env.close()

def parse_args():
    ap = argparse.ArgumentParser('Evaluate Parser')
    ap.add_argument('--model_path', type=str, default='best_reward_model.pth',
                    help="Path to the model .pth file")
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help="Device to use")
    ap.add_argument('--mode', type=str, default='human', choices=['human', 'ai'],
                    help="Choose 'human' to play via keyboard against the Atari AI, or 'ai' to evaluate the model.")
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)