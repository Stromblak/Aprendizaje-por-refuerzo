# conda create -c conda-forge -n gymenv swig pip 
# conda activate gymenv

# pip install panda-gym stable-baselines3 tensorboard sb3-contrib
# pip install swig Box2D

import gym
import panda_gym
import math
import time

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO, TD3, DDPG, SAC, A2C
from sb3_contrib import TRPO, TQC

from stable_baselines3.common.env_util import make_vec_env


# ambiente
ambiente = "BipedalWalker-v3"
env = make_vec_env(ambiente, n_envs=1)


# entrenamiento
pablo = [PPO, DDPG]
dazhi = [A2C, TD3, SAC]
jose = [TRPO, TQC]

for algo in pablo + dazhi + jose:
    carpeta = str(math.floor(time.time())) + "/" 
    model = algo("MlpPolicy", # "MultiInputPolicy"
                env, 
                verbose=1, 
                device="cuda", 
                tensorboard_log=carpeta)

    model.learn(total_timesteps=500_000)
    model.save(carpeta + ambiente)


# Ejecutar el modelo entrenado en el entorno
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")