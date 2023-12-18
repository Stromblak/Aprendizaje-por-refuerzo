import gymnasium as gym
from tarea2v2 import DQNAgent, wrap_env


env = gym.make("ALE/Frogger-v5", render_mode="human") #human
env = wrap_env(env)
env.reset()

agent = DQNAgent(env)
agent.load_params("pesos")


for episode in range(10):
    while True:
        if agent.play(0):
            break
