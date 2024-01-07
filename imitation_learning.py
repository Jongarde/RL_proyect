import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from huggingface_sb3 import load_from_hub
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer

import numpy as np
import pickle

# Random number generator for the experiments
rng=np.random.default_rng(0)

# Trains an expert using PPO and returns the model
def train_expert(env, total_timesteps=1_000_000):
    # SB3-Zoo version
    #algo = PPO(policy="MlpPolicy", env=env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, verbose=True)
    # Inaki Vazquez version
    algo = PPO('MlpPolicy', env=env, learning_rate=0.002, batch_size=64, policy_kwargs=dict(net_arch=[8,32]), verbose=True)    

    algo.learn(total_timesteps=total_timesteps)
    return algo

def use_huggingface_expert(huggingface_repo_id, huggingface_filename):
    checkpoint = load_from_hub(
        repo_id=huggingface_repo_id,
        filename=huggingface_filename,
    )
    expert = PPO.load(checkpoint)
    return expert

# Executes episodes with the expert returning the transitions
def collect_expert_transitions(expert,env):
    rollouts = rollout.rollout(expert, DummyVecEnv([lambda: RolloutInfoWrapper(env)]), rollout.make_min_episodes(n=100), rng=rng)
    return rollout.flatten_trajectories(rollouts)

# Saves the transitions in a file
def save_expert_transitions(transitions, filename="expert_transitions.pickle"):
    fileh = open(filename, "wb")
    pickle.dump(transitions, fileh)

# Loads the transitions from a file
def load_expert_transitions(filename="expert_transitions.pickle"):
    fileh = open(filename, "rb")
    return pickle.load(fileh)

# Trains a model based on the transitions using Behavioural Cloning
def train_bc(env, transitions, epochs=4):
    bc_trainer = BC(observation_space=env.observation_space, action_space=env.action_space, demonstrations=transitions, rng=rng)
    bc_trainer.train(n_epochs=epochs)
    return bc_trainer.policy

# Trains a model using DAgger
def train_dagger(env, n_steps, expert):
    checkpoint_dir="dagger_temp"
    os.system("rm -rf ./" + checkpoint_dir) # clean from previous executions, remove this line to continue incremental training
    venv = DummyVecEnv([lambda: env]) # dagger requires a vectorized environment
    bc_trainer = BC(observation_space=env.observation_space, action_space=env.action_space, rng=np.random.default_rng(0))
    dagger_trainer = SimpleDAggerTrainer(venv=venv, scratch_dir=checkpoint_dir, expert_policy=expert, bc_trainer=bc_trainer, rng=rng)
    dagger_trainer.train(n_steps)
    return dagger_trainer.policy

# Trains an expert and saves the trasitions
def generate_expert_and_transitions(env_name):
    env = gym.make(env_name, render_mode=None)
    env = Monitor(env)
    #expert = train_expert(env=env, total_timesteps=150_000)
    expert = use_huggingface_expert("MadFritz/ppo-BipedalWalker-v3", "ppo-BipedalWalker-v3.zip")
    expert.save("ppo_bipedal_expert")
    print("Collecting expert transitions...")
    transitions = collect_expert_transitions(expert, env)
    save_expert_transitions(transitions)
    avg_reward, _ = evaluate_policy(expert, env, n_eval_episodes=10, render=False)
    print ("Avg reward: ", avg_reward)

# Compares the expert with BC and DAgger
def run_experiments(env_name):
    env_train = gym.make(env_name, render_mode=None)
    env_train = Monitor(env_train)

    env_eval = gym.make(env_name, render_mode="human")
    env_eval = Monitor(env_eval)

    print("Testing the expert...")
    expert = PPO.load("ppo_bipedal_expert")
    expert_avg_reward, _ = evaluate_policy(expert, env_eval, n_eval_episodes=10, render=True)

    transitions = load_expert_transitions()
    bc_algo = train_bc(env_train, transitions)
    print("Testing behavioural cloning...")
    bc_avg_reward, _ = evaluate_policy(bc_algo, env_eval, n_eval_episodes=10, render=True)

    dagger_algo = train_dagger(env_train, 10_000, expert=expert)
    print("Testing DAgger...")
    dagger_avg_reward, _ = evaluate_policy(dagger_algo, env_eval, n_eval_episodes=10, render=True)

    print ("Expert - Avg reward: ", expert_avg_reward)
    print ("BC - Avg reward: ", bc_avg_reward)
    print ("DAgger - Avg reward: ", dagger_avg_reward)

environment_name = 'BipedalWalker-v3' 

# uncomment the next line for generting the expert policy and trasitions file
generate_expert_and_transitions(environment_name)
run_experiments(environment_name)
