import gymnasium as gym
gym.logger.set_level(40)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import math
import time
import cv2
import os

from torch.utils.tensorboard import SummaryWriter



# Codigo obtenido de trabajos pasados del equipo


# procesamiento de imagen
class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(105, 80, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        
        resized_screen = cv2.resize(img, (80, 105), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [105, 80, 1])

        #sinResize = np.reshape(img, [210, 160, 1]).astype(np.uint8)        
        return x_t.astype(np.uint8)
    
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def wrap_env(env):
    env = ProcessFrame(env)
    env = ImageToPyTorch(env)
    env = ScaledFloatFrame(env)
    return env


# Replay Memory
import collections
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state', 'done'])
class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idxs])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
    

# Red neuronal 
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# Agente
class DQNAgent:
    def __init__(self, env, render = True):
        self.env = env
        self.replay_buffer = ReplayMemory(max_size=BUFFER_SIZE)
        self.reset()
        self.render = render

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()	# para que no actualize los pesos el optimizador ?

        # self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr = 1e-4 )
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()
    
    def reset(self):
        self.total_reward = 0.0
        self.vidaAnterior = 4
        self.state, info = self.env.reset()
        
    def play(self, eps):
        # epsilon-greedy
        if np.random.random() < eps:
            action = self.env.action_space.sample()
            #action = random.choice([0, 1, 2])

        else:
            state_tensor = torch.FloatTensor(self.state.astype(np.float32)).unsqueeze(0).to(self.device)
            qvals_tensor = self.policy(state_tensor)
            action = int(np.argmax(qvals_tensor.cpu().detach().numpy()))


        next_state, reward, done, _, info = self.env.step(action) #4 if action == 2 else action

        if action == 4:
            reward -= 0.01
        elif action == 1:
            reward += 0.01

        if info["lives"] < self.vidaAnterior:
            reward = -0.5
        
        self.vidaAnterior = info["lives"]


        experience = Experience(self.state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
        self.state = next_state
        self.total_reward += reward
        

        if self.render:
            self.env.render()

        if done:
            res_reward = self.total_reward
            self.reset()
            return res_reward
        else:
            return None
    
    def optimizarModelo(self):
        if len(self.replay_buffer) < REPLAY_START:
            return

        # batch de experiencia y optimizacion de los datos
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states_tensor 		= torch.tensor(states.astype(np.float32), dtype=torch.float32).to(self.device)
        actions_tensor 		= torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor 		= torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor 	= torch.tensor(next_states.astype(np.float32), dtype=torch.float32).to(self.device)
        dones_tensor 		= torch.tensor(dones, dtype=torch.int).to(self.device)

        # DQN
        curr_Q = self.policy(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            max_next_Q = self.target(next_states_tensor).max(1)[0]
            max_next_Q *= 1 - dones_tensor

        expected_Q = rewards_tensor + GAMMA * max_next_Q.detach()

        loss = self.loss_fn(curr_Q, expected_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target()

    def update_target(self):
        for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)	

    def load_params(self, file_name):
        print(self.policy.load_state_dict(torch.load(file_name)))

        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()


# Bucle entrenamiento
def entrenamiento(agent: DQNAgent):
    episode_rewards = []
    eps = EPS_START
    total_steps = 0
    best_mean_reward = 0
    
    carpeta = str(math.floor(time.time())) + "/"
    writer = SummaryWriter(carpeta)
    last_DQN = ""
    
    for episode in range(EPISODES):
        # evaluacion
        if episode%20 == 0 and episode > 0:
            recompensasGreedy = []
            for _ in range(10):
                while True:
                    episode_reward = agent.play(0)

                    if episode_reward is not None:
                        recompensasGreedy.append(episode_reward)
                        break
            
            writer.add_scalar("reward_greedy", np.mean(recompensasGreedy), int(episode/20))

        # guardar modelo
        if (episode+1)%100 == 0 and episode > 0:
            torch.save(agent.policy.state_dict(), carpeta + "DQN_" + str(episode+1) + "_" + str(mean_reward))

        # entrenamiento real
        while True:
            total_steps += 1
            
            eps = max(eps*EPS_DECAY, EPS_END)

            episode_reward = agent.play(eps)

            # si episode_reward = None entonces no ha terminado
            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                mean_reward = np.mean(episode_rewards[-100:])

                if mean_reward > best_mean_reward and episode >= 50:
                    best_mean_reward = mean_reward

                    if os.path.exists(last_DQN):
                        os.remove(last_DQN)

                    last_DQN = carpeta + "DQN_" + str(mean_reward)
                    torch.save(agent.policy.state_dict(), last_DQN)
                    print("Best mean reward updated, model saved")
                    

                print("Episodio: %d | mean reward %.3f | steps: %d | epsilon %.3f" % (episode, mean_reward, total_steps,  eps))
                break
            
            agent.optimizarModelo()

        writer.add_scalar("episode_reward", episode_reward, episode)
        writer.add_scalar("mean_reward", mean_reward, episode)
        writer.add_scalar("epsilon", eps, episode)


    writer.close()
    return episode_rewards


# ----------------------------------- Hiperparametros ------------------------------------
BATCH_SIZE          = 128				# numero de experiencias del buffer a usar
REPLAY_START        = BATCH_SIZE	    # steps minimos para empezar	
BUFFER_SIZE  		= 100000			# steps maximos que guarda 

TAU 				= 0.01				# actualizacion pasiva de la target network

EPISODES			= 10000   
GAMMA				= 0.90
EPS_START			= 1
EPS_END				= 0.05
EPS_DECAY 			= 0.99996
LEARNING_RATE		= 0.00025 		# 1e-4



def main():
    env = gym.make("ALE/Frogger-v5", render_mode="rgb_array") #human
    env = wrap_env(env)
    env.reset()

    agent = DQNAgent(env, render=True)
    entrenamiento(agent)


if __name__ == "__main__":
   main()