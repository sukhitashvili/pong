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

# import argparse
# import cv2
# import ale_py
# import gymnasium as gym
# import numpy as np
# import torch

# from utils import display_observation, image_preprocess

# @torch.no_grad()
# def main(args):
#     # init the parameters
#     display = True
#     D = 80 * 80  # input dimensionality: 80x80 grid
#     prev_x = None
#     reward_sum = 0

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Load model only if evaluating the AI
#     if args.mode == "ai":
#         model = torch.load(
#             args.model_path,
#             map_location=device,
#             weights_only=False
#         )
#         model.eval()
#         model.to(device)
#     else:
#         print("=====================================================")
#         print("HUMAN MODE SELECTED")
#         print("Control the right paddle using your KEYBOARD.")
#         print(" - Press 'W' to move UP")
#         print(" - Press 'S' to move DOWN")
#         print(" - Press 'Q' to QUIT")
#         print("Make sure the OpenCV game window is in focus!")

#     # init the game
#     env = gym.make("Pong-v4", render_mode="rgb_array")
#     observation, info = env.reset(seed=42)

#     # Initialize OpenCV window to capture keyboard input
#     cv2.namedWindow("Pong")

#     while True:
#         if args.mode == "ai":
#             # preprocess the observation, set input to network to be difference image
#             cur_x = image_preprocess(observation, device=device)
#             input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D, device=device)
#             prev_x = cur_x

#             # model forward pass
#             output = model(input_x)
#             action = 2 if np.random.uniform() < output.item() else 3  # roll the dice!
            
#             # Allow user to quit early
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#         else:
#             # Take user input via KEYBOARD
#             # A 30ms wait provides a playable ~33 FPS for the human
#             key = cv2.waitKey(30) & 0xFF
            
#             if key == ord('w'):
#                 action = 2  # UP
#             elif key == ord('s'):
#                 action = 3  # DOWN
#             elif key == ord('q'):
#                 break       # QUIT
#             else:
#                 action = 0  # NOOP

#         # step the environment and get new measurements
#         observation, reward, terminated, truncated, info = env.step(action)
#         reward_sum += reward
        
#         # display game if needed
#         if display:
#             # Convert RGB array from Gymnasium to BGR for OpenCV
#             #bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
#             #cv2.imshow("Pong", bgr_image)

#             bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

#             # Scale by 4x
#             display_image = cv2.resize(
#             bgr_image,
#             None,
#             fx=4,
#             fy=4,
#             interpolation=cv2.INTER_NEAREST
#         )

#         cv2.imshow("Pong", display_image)

#         if terminated or truncated:  # an episode finished, someone reached 21 scores
#             print('Episode total reward:', reward_sum)
#             break

#     cv2.destroyAllWindows()
#     env.close()

# def parse_args():
#     ap = argparse.ArgumentParser('Evaluate Parser')
#     ap.add_argument('--model_path', type=str, default='best_reward_model.pth',
#                     help="Path to the model .pth file")
#     ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
#                     help="Device to use")
#     ap.add_argument('--mode', type=str, default='human', choices=['human', 'ai'],
#                     help="Choose 'human' to play via keyboard against the Atari AI, or 'ai' to evaluate the model.")
#     args = ap.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

import argparse
import cv2
import ale_py
import gymnasium as gym
import numpy as np
import torch
import time

from utils import display_observation, image_preprocess

def draw_tennis_racket(img, cx, cy, is_left_player, scale):
    """Draws a tennis racket at the specified center coordinates."""
    # Racket dimensions based on scale
    handle_width = 2 * scale
    handle_length = 8 * scale
    head_width = 6 * scale
    head_height = 9 * scale
    
    # Colors (BGR for OpenCV)
    handle_color = (20, 60, 100)  # Brownish handle
    frame_color = (180, 180, 180) # Silver frame
    string_color = (50, 50, 50)   # Dark string grid
    
    if is_left_player:
        # Handle pointing left
        cv2.rectangle(img, (cx - head_width//2 - handle_length, cy - handle_width//2),
                      (cx - head_width//2, cy + handle_width//2), handle_color, -1)
    else:
        # Handle pointing right
        cv2.rectangle(img, (cx + head_width//2, cy - handle_width//2),
                      (cx + head_width//2 + handle_length, cy + handle_width//2), handle_color, -1)
        
    # Draw strings (grid)
    for i in range(-head_width//2 + 2, head_width//2, 3):
        cv2.line(img, (cx + i, cy - head_height//2 + 2), (cx + i, cy + head_height//2 - 2), string_color, 1)
    for i in range(-head_height//2 + 2, head_height//2, 3):
        cv2.line(img, (cx - head_width//2 + 2, cy + i), (cx + head_width//2 - 2, cy + i), string_color, 1)

    # Draw racket head frame
    cv2.ellipse(img, (cx, cy), (head_width//2, head_height//2), 0, 0, 360, frame_color, 2, cv2.LINE_AA)

def draw_tennis_ball(img, cx, cy, scale):
    """Draws a yellow tennis ball with white seams."""
    radius = 2 * scale
    # Draw yellow ball
    cv2.circle(img, (cx, cy), radius, (0, 220, 220), -1, cv2.LINE_AA)
    # Draw white seams (approximated with arcs)
    cv2.ellipse(img, (cx - radius//2, cy), (radius//2, int(radius*0.8)), 0, -60, 60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + radius//2, cy), (radius//2, int(radius*0.8)), 0, 120, 240, (255, 255, 255), 1, cv2.LINE_AA)

@torch.no_grad()
def main(args):
    # init the parameters
    display = True
    D = 80 * 80  # input dimensionality: 80x80 grid
    prev_x = None
    reward_sum = 0
    duration_seconds = 300  # 5 minutes

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
        print("🕹️ HUMAN MODE SELECTED")
        print("Control the right paddle using your KEYBOARD.")
        print(" - Press 'W' to move UP")
        print(" - Press 'S' to move DOWN")
        print(" - Press 'Q' to QUIT")
        print(f"Game will run continuously for {duration_seconds // 60} minutes.")
        print("Make sure the OpenCV game window is in focus!")
        print("=====================================================")

    # init the game
    env = gym.make("Pong-v4", render_mode="rgb_array")
    observation, info = env.reset(seed=42)

    cv2.namedWindow("Pong Table Tennis Pro")
    
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        if args.mode == "ai":
            cur_x = image_preprocess(observation, device=device)
            input_x = cur_x - prev_x if prev_x is not None else torch.zeros(D, device=device)
            prev_x = cur_x

            output = model(input_x)
            action = 2 if np.random.uniform() < output.item() else 3  
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('w'): action = 2 
            elif key == ord('s'): action = 3 
            elif key == ord('q'): break     
            else: action = 0 

        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        
        if display:
            bgr_image = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            scale = 4
            h, w = bgr_image.shape[:2]
            
            # 1. Create a blank canvas scaled up by 4x
            custom_frame = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
            
            # 2. Preserve the original score area at the top
            score_area = cv2.resize(bgr_image[0:34, :], (w * scale, 34 * scale), interpolation=cv2.INTER_NEAREST)
            custom_frame[0:34*scale, :] = score_area
            
            # 3. Draw the Green Table Tennis Court background
            cv2.rectangle(custom_frame, (0, 34*scale), (w*scale, h*scale), (60, 140, 60), -1) 
            cv2.rectangle(custom_frame, (10*scale, 40*scale), (w*scale - 10*scale, h*scale - 5*scale), (255, 255, 255), max(1, scale//2))
            cv2.line(custom_frame, (w*scale//2, 40*scale), (w*scale//2, h*scale - 5*scale), (200, 200, 200), max(1, scale//2))
            
            # 4. Locate the objects using color masking
            play_area = bgr_image[34:, :] 
            
            # Detect Left Paddle
            left_mask = cv2.inRange(play_area, np.array([50, 100, 180]), np.array([100, 160, 240]))
            contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                lx, ly, lw, lh = cv2.boundingRect(cnt)
                if lh > 5: # Valid paddle
                    draw_tennis_racket(custom_frame, (lx + lw//2) * scale, (ly + 34 + lh//2) * scale, is_left_player=True, scale=scale)

            # Detect Right Paddle
            right_mask = cv2.inRange(play_area, np.array([70, 150, 70]), np.array([120, 210, 120]))
            contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rh > 5: # Valid paddle
                    draw_tennis_racket(custom_frame, (rx + rw//2) * scale, (ry + 34 + rh//2) * scale, is_left_player=False, scale=scale)

            # Detect Ball using Contours instead of flat bounding box
            ball_mask = cv2.inRange(play_area, np.array([200, 200, 200]), np.array([255, 255, 255]))
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Ignore the center net pixels
                if not (78 <= bx <= 82):
                    if bw < 5 and bh < 5: # Ball is tiny
                        draw_tennis_ball(custom_frame, (bx + bw//2) * scale, (by + 34 + bh//2) * scale, scale=scale)

            cv2.imshow("Pong Table Tennis Pro", custom_frame)

        # Reset instead of breaking when the game ends
        if terminated or truncated:
            print(f"Match concluded! Reward sum: {reward_sum}. Resetting for next match...")
            observation, info = env.reset()
            prev_x = None  # Crucial to reset AI motion frame
            reward_sum = 0

    print("5 minutes elapsed. Exiting game.")
    cv2.destroyAllWindows()
    env.close()

def parse_args():
    ap = argparse.ArgumentParser('Evaluate Parser')
    ap.add_argument('--model_path', type=str, default='best_reward_model.pth', help="Path to the model")
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--mode', type=str, default='human', choices=['human', 'ai'])
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
