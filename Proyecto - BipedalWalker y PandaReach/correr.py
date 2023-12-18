# conda create -c conda-forge -n gymenv swig pip 
# conda activate gymenv

# pip install panda-gym stable-baselines3 tensorboard sb3-contrib
# pip install swig Box2D

import gym
import panda_gym
import math
import time
import sys

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO, TD3, DDPG, SAC, A2C
from sb3_contrib import TRPO, TQC

from stable_baselines3.common.env_util import make_vec_env


if sys.argv[1] == "p":
    env = make_vec_env("PandaReach-v3", n_envs=1)
    obs = env.observation_space
    act = env.action_space
    model = SAC.load("PandaReach/1702421039/PandaReach-v3.zip", env,
                     custom_objects={'observation_space': obs, 'action_space': act})

elif sys.argv[1] == "w":
    env = make_vec_env("BipedalWalker-v3", n_envs=1)
    obs = env.observation_space
    act = env.action_space
    model = SAC.load("Walker/1702442342/BipedalWalker-v3.zip", env,
                     custom_objects={'observation_space': obs, 'action_space': act})



# Ejecutar el modelo entrenado en el entorno
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")