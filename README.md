# About

A Pytorch reimplementation of Andrej Karpathy's blog [Deep Reinforcement Learning: Pong from Pixels.](https://karpathy.github.io/2016/05/31/rl/)
The RL agent learns to play Pong via trial and error from pixels, using Policy Gradient RL method implemented in PyTorch.

The training code is located at `main.py`, and the slightly modified original Andrej's code is stored at `karpathys_code.py` (it was adapted for Python 3.10).


![My Image](images/screenshot.png)


# Installation

Requirements:
  - Python 3.10+
  - CUDA Version: 12 (if CUDA is not available, comment out Nvidia-related packages before installing `requirements.txt` to train on CPU)
  - Install requirements `pip install -r requirements.txt`

# Train

To train a new model, run: `python3.10 main.py`.
Check Andrej Karpathy's [Blog](https://karpathy.github.io/2016/05/31/rl/) for more details of the training, algorithm, etc.

# Play

To play, run: `python3.10 play.py --model_path=best_reward_model.pth --device="cuda:0"`

# Model Learning Curve

One episode consists of 21 games, and each point represents an exponentially weighted average
across played episodes. The graph shows that the model
gradually learns to play the game, and rewards reach positive values after around 
4000 episodes. 

Learning slows down after ~4000 episodes, since players play the game equally well. 
This prolongs games and, thus, they are truncated by the Gymnasium library, resulting 
model receiving a zero reward. This explains why learning slows down after ~4000 episodes.

![My Image](images/plot.png)
