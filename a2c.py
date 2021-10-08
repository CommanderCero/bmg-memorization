import utils

import numpy as np
import gym
import gym_ptb
import wandb
import datetime
import os

from pathlib import Path
from timeit import default_timer as timer
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(actor_critic, env_fn: Callable[[], gym.Env],
          train_steps=1000, env_count=1, env_steps=128, reward_decay=0.99,
          device = torch.device('cpu'), lr=0.0003,
          entropy_weight=0.01, max_gradient_norm=0.5,
          save_frequency=50, log_folder='.', log_video=False):
    
    # Initialize environment and buffer
    env = gym.vector.AsyncVectorEnv([env_fn] * env_count)
    buffer = utils.VecTrajectoryBuffer(
        env.single_observation_space.shape,
        env.single_action_space.shape,
        env_count, env_steps)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    # Training
    start = timer()
    episode_lengths = [[0] for _ in range(env_count)]
    episode_returns = [[0] for _ in range(env_count)]
    log_step = 0
    states = env.reset()
    for step in range(train_steps):
        ### Collect training data
        buffer.clear()
        for _ in range(env_steps):
            inp = torch.as_tensor(states, dtype=torch.float32).to(device)
            actions = actor_critic.get_actions(inp)
            new_states, rewards, dones, _ = env.step(actions)
            
            # Store data
            buffer.store(states, actions, rewards, dones)
            states = new_states
            
            # Keep track of episode statistics
            for i, done in enumerate(dones):
                episode_lengths[i][-1] += 1
                episode_returns[i][-1] += rewards[i]
                if done:
                    episode_lengths[i].append(0)
                    episode_returns[i].append(0)
        # Bootstrap the value for the last visited state
        with torch.no_grad():
            inp = torch.Tensor(states).to(device)
            values = actor_critic.get_values(inp)
            buffer.end_trajectory(values)
        
        ### Train
        data = buffer.get_data(device)
        data["returns"] -= data["returns"].mean()
        data["returns"] /= data["returns"].std() + 1e-6
        dist, values = actor_critic(data["observations"])
        values = values.reshape(-1)
        
        # Compute Actor "Loss"
        log_probs = actor_critic.get_log_probs(dist, data["actions"])
        advantages = data["returns"] - values.detach()
        actor_loss = -torch.mean(advantages * log_probs)
        
        # Compute Critic Loss
        critic_loss = F.mse_loss(values, data["returns"])
        
        # Train critic and actor -> Note we subtract entropy to encourage exploration
        entropy_loss = -torch.mean(dist.entropy())
        loss = actor_loss + critic_loss + entropy_loss * entropy_weight
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_gradient_norm)
        optimizer.step()
        
        ### Logging
        if step % save_frequency == 0 or step == train_steps - 1:
            # Flatten the episode stats and ignore the last entry since it's an unfinished episode
            flat_returns = [val for arr in episode_returns for val in arr]
            flat_lengths = [val for arr in episode_lengths for val in arr]
            print(f"{step}: Avg Return={np.mean(flat_returns)}")
            
            # Save networks
            torch.save(actor_critic.state_dict(), os.path.join(log_folder, "actor_latest.torch"))
            
            # Weights & Biases logging
            wandb.log({
                 "Actor Loss": actor_loss,
                 "Critic Loss": critic_loss,
                 "Entropy Loss": entropy_loss,
                 "Avg. Return": np.mean(flat_returns), 
                 "Max Return": np.max(flat_returns),
                 "Min Return": np.min(flat_returns),
                 "Avg. Episode Length": np.mean(flat_lengths),
                 "Time Passed (Minutes)": (timer() - start) / 60},
                 step=log_step)
            
            #if log_video:
            #    wandb.log({"Actor": utils.capture_video(actor, env_fn(), device=device)}, step=log_step)
            
            params = {f"actor-critic/param/{name}" : wandb.Histogram(param.detach().cpu()) for name, param in actor_critic.named_parameters()}
            grads = {f"actor-critic/gradient/{name}" : wandb.Histogram(param.grad.cpu()) for name, param in actor_critic.named_parameters()}
            wandb.log(params, step=log_step)
            wandb.log(grads, step=log_step)
            
            # Clear old logging data but keep the unfinished episode
            episode_lengths = [[episode_lengths[i][-1]] for i in range(env_count)]
            episode_returns = [[episode_returns[i][-1]] for i in range(env_count)]
            
            # Advance log_step
            log_step += 1

class PTBActorCritic(nn.Module):
    def __init__(self, embedding_size, layers, obs_shape, num_actions):
        super().__init__()
        
        self.embedding = nn.Embedding(256, embedding_size)
        obs_size = np.prod(obs_shape) * embedding_size
        self.actor_body = utils.create_feedforward([obs_size, *layers, num_actions])
        self.critic_body = utils.create_feedforward([obs_size, *layers, 1])
        
    def forward(self, X):
        embedded_X = self.embedding(X.long()).reshape(X.shape[0], -1)
        actor_logits = self.actor_body(embedded_X)
        values = self.critic_body(embedded_X)
        return torch.distributions.Categorical(logits=actor_logits), values
    
    
    @torch.no_grad()
    def get_values(self, X: torch.Tensor) -> torch.Tensor:
        embedded_X = self.embedding(X.long()).reshape(X.shape[0], -1)
        values = self.critic_body(embedded_X)
        return values.cpu().numpy()
    
    @torch.no_grad()
    def get_actions(self, X: torch.Tensor) -> torch.Tensor:
        embedded_X = self.embedding(X.long()).reshape(X.shape[0], -1)
        actor_logits = self.actor_body(embedded_X)
        dist = torch.distributions.Categorical(logits=actor_logits)
        return dist.sample().cpu().numpy()
    
    def get_log_probs(self, dist: torch.distributions.Distribution, actions: torch.Tensor) -> torch.Tensor:
        return dist.log_prob(actions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--embedding_size', type=int, default=4)
    parser.add_argument('--rew_decay', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_count', type=int, default=2, help='The number of environments that are executed in paralel. The batch_size equals env_count * env_steps.')
    parser.add_argument('--env_steps', type=int, default=128, help='The number of steps to take for each parallel environment. The batch_size equals env_count * env_steps.')
    parser.add_argument('--steps', type=int, default=1000, help='The total number of gradient-updates until the training is considered complete.')
    parser.add_argument('--exp_name', type=str, default='press_the_button')
    parser.add_argument('--log_folder', type=str, default='./logs')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--log_video', type=bool, default=False)
    args = parser.parse_args()
    
    # Set seed for deterministic results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Gather information about environment
    env_fn = lambda: gym.make('PressTheButton-v0')
    dummy_env = env_fn()
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    del dummy_env
    
    # Initialize actor-critic
    actor_critic = PTBActorCritic(args.embedding_size, args.layers, obs_space.shape, act_space.n)
    
    # Setup Logging
    full_experiment_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_folder = os.path.join(args.log_folder, full_experiment_name)
    print(f"Log Folder '{log_folder}'")
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    wandb.init(name=full_experiment_name, project='RL-Algorithms', dir=log_folder)
    
    # Train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    actor_critic.to(device)
    
    print("---- STARTING TRAINING ----")
    train(actor_critic, env_fn, 
          train_steps=args.steps,
          device=device,
          env_count=args.env_count, env_steps=args.env_steps,
          save_frequency=args.save_freq,
          reward_decay=args.rew_decay,
          log_folder=log_folder, log_video=args.log_video)