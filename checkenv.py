import gymnasium as gym
import random
import os
import numpy as np
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

ALGORITHM = "PPO"
models_dir = "models/" + ALGORITHM
log_dir = "logs"		

train = False

ENV_ID = "BipedalWalker-v3"
DEFAULT_HYPERPARAMS = {
	"policy": "MlpPolicy",
	"env": ENV_ID,
}

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

if train:
    env = gym.make("BipedalWalker-v3")
else:
    env = gym.make("BipedalWalker-v3", render_mode = "human")

env = DummyVecEnv([lambda: env])

TIMESTEPS = 10000000

if train:
	with open('optuna_best_trial.json', 'r') as json_file:
		data = json.load(json_file)

	net_arch = {
	"tiny": dict(pi=[64], vf=[64]),
	"small": dict(pi=[64, 64], vf=[64, 64]),
	"medium": dict(pi=[256, 256], vf=[256, 256]),
	}[data['net_arch']]

	activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[data['activation_fn']]
		
	policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn, ortho_init=False)	

	data.pop('activation_fn')
	data.pop('net_arch')

	kwargs = DEFAULT_HYPERPARAMS.copy()
	kwargs.update(data)
	
	env.reset()

	model = PPO(**kwargs, policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
	"""
	if os.path.exists(f"{models_dir}/{TIMESTEPS}.zip"):
		model = PPO.load(f"{models_dir}/{TIMESTEPS}", env=env)
	"""

	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=ALGORITHM, progress_bar=True)
	model.save(f"{models_dir}/{TIMESTEPS}-2")

else:
	model = PPO.load(f"{models_dir}/{TIMESTEPS}-2", env=env)
	obs = env.reset()
	while True:
		action, _states = model.predict(obs, deterministic=True)
		env.render()
		obs, reward, done, info = env.step(action)
		if done:
			aobs = env.reset()
	env.close()

