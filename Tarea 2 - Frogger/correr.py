import gymnasium as gym
from tarea2v2 import DQNAgent, wrap_env

import numpy as np

env = gym.make("ALE/Frogger-v5", render_mode="rgb_array") #human
env = wrap_env(env)
env.reset()
agent = DQNAgent(env)


import os
directorio = os.path.join(os.path.dirname(__file__), "1702155263")

# Obtén la lista de archivos en el directorio
archivos = os.listdir(directorio)



for a in archivos:
    agent.reset()
    agent.load_params("1702155263/" + a)
    
    print(a)
    recompensas = []
    for episode in range(40):
        print(episode)
        while True:
            episode_reward = agent.play(0)
            if episode_reward is not None:
                recompensas.append(episode_reward)
                break
                

    print(np.mean(recompensas))
    print()
