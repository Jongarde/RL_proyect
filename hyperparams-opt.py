import os
import json
import gymnasium as gym
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

ENV_ID = "BipedalWalker-v3"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}

def sample_ppo_params(trial: optuna.Trial):
	batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
	n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
	gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
	learning_rate = trial.suggest_float("learning_rate", 5e-6, 0.003, log=True)
	ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.01, log=True)
	clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
	n_epochs = trial.suggest_categorical("n_epochs", [3, 5, 10, 15, 20, 25, 30])
	gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
	max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
	vf_coef = trial.suggest_float("vf_coef", 0.5, 1)
	net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
	ortho_init = False
	activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
	if batch_size > n_steps:
		batch_size = n_steps

	net_arch = {
	"tiny": dict(pi=[64], vf=[64]),
	"small": dict(pi=[64, 64], vf=[64, 64]),
	"medium": dict(pi=[256, 256], vf=[256, 256]),
	}[net_arch_type]

	activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

	return {
	"n_steps": n_steps,
	"batch_size": batch_size,
	"gamma": gamma,
	"learning_rate": learning_rate,
	"ent_coef": ent_coef,
	"clip_range": clip_range,
	"n_epochs": n_epochs,
	"gae_lambda": gae_lambda,
	"max_grad_norm": max_grad_norm,
	"vf_coef": vf_coef,
	# "sde_sample_freq": sde_sample_freq,
	"policy_kwargs": dict(
	# log_std_init=log_std_init,
	net_arch=net_arch,
	activation_fn=activation_fn,
	ortho_init=ortho_init,
	),
	}


def evaluate_model(model, env, num_episodes=5):
	total_reward = 0.0
	for _ in range(num_episodes):
		obs = env.reset()
		done = False
		while not done:
			action, _ = model.predict(obs)
			obs, reward, done, _ = env.step(action)
			total_reward += reward
	mean_reward = total_reward / num_episodes
	return mean_reward


def objective(trial):
	kwargs = DEFAULT_HYPERPARAMS.copy()
	kwargs.update(sample_ppo_params(trial))
	
	env = DummyVecEnv([lambda: gym.make('BipedalWalker-v3')])
	
	model = PPO(**kwargs)

	# Train for a certain number of steps
	total_timesteps = 100000  # Adjust based on your needs
	try:
		model.learn(total_timesteps=total_timesteps, progress_bar=True)

		# Evaluate the trained model
		mean_reward = evaluate_model(model, env, num_episodes=5)
		return mean_reward
	except:
		return -200


if __name__ == "__main__":
	study = optuna.create_study(direction="maximize")
	
	study.optimize(objective, n_trials=50)
	
	fig = optuna.visualization.plot_optimization_history(study)
	fig.show()
	fig = optuna.visualization.plot_contour(study)
	fig.show()
	fig = optuna.visualization.plot_slice(study)
	fig.show()
	fig = optuna.visualization.plot_param_importances(study)
	fig.show()

	best_trial = study.best_trial
	print("Best trial:")
	print("  Value: ", best_trial.value)
	print("  Params: ")
	best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
	print(best_trial_params)

	best_trial_file = open("optuna_best_trial.json", "w")
	best_trial_file.write(best_trial_params)
	best_trial_file.close()



