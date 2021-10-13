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

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(actor_critic, env_fn: Callable[[], gym.Env],
          train_steps=1000, env_steps=128, reward_decay=0.99,
          device = torch.device('cpu'), lr=0.0003,
          entropy_weight=0.01, max_gradient_norm=0.5,
          save_frequency=50, log_folder='.', log_video=False):
    
    # Initialize environment and buffer
    env = env_fn()
    buffer = utils.TrajectoryBuffer(
        env.observation_space.shape,
        env.action_space.shape,
        env_steps)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    # Training
    start = timer()
    episode_lengths = [0]
    episode_returns = [0]
    log_step = 0
    state = env.reset()
    for step in range(train_steps):
        ### Collect training data
        values = []
        log_probs = []
        entropy = []
        buffer.clear()
        for _ in range(env_steps):
            # Advance environment
            inp = torch.as_tensor(state, dtype=torch.float32).to(device)
            dist, value = actor_critic(inp.unsqueeze(0))
            action = dist.sample()
            new_state, reward, done, _ = env.step(action.item())
            
            # Store data
            values.append(value)
            log_probs.append(dist.log_prob(action))
            entropy.append(dist.entropy())
            buffer.store(state, action.item(), reward, done)
            state = new_state
            
            # Keep track of episode statistics
            episode_lengths[-1] += 1
            episode_returns[-1] += reward
            if done:
                episode_lengths.append(0)
                episode_returns.append(0)
                
                # Reset agents hidden state
                state = env.reset()
                actor_critic.reset()
        # Bootstrap the value for the last visited state
        with torch.no_grad():
            inp = torch.Tensor(state).to(device)
            value = actor_critic.get_values(inp.unsqueeze(0)).item()
            buffer.end_trajectory(value)
        
        ### Train
        data = buffer.get_data(device)
        data["returns"] -= data["returns"].mean()
        data["returns"] /= data["returns"].std() + 1e-6
        
        # Compute Actor "Loss"
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        advantages = data["returns"] - values.detach()
        actor_loss = -torch.mean(advantages * log_probs)
        
        # Compute Critic Loss
        critic_loss = F.mse_loss(values, data["returns"])
        
        # Train critic and actor -> Note we subtract entropy to encourage exploration
        entropy_loss = -torch.mean(torch.cat(entropy))
        loss = actor_loss + critic_loss + entropy_loss * entropy_weight
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_gradient_norm)
        optimizer.step()
        
        actor_critic.reset()
        
        ### Logging
        if step % save_frequency == 0 or step == train_steps - 1:
            print(f"{step}: Avg Return={np.mean(episode_returns)}")
            
            # Save networks
            torch.save(actor_critic.state_dict(), os.path.join(log_folder, "actor_latest.torch"))
            
            # Weights & Biases logging
            wandb.log({
                 "Actor Loss": actor_loss,
                 "Critic Loss": critic_loss,
                 "Entropy Loss": entropy_loss,
                 "Avg. Return": np.mean(episode_returns), 
                 "Max Return": np.max(episode_returns),
                 "Min Return": np.min(episode_returns),
                 "Avg. Episode Length": np.mean(episode_lengths),
                 "Time Passed (Minutes)": (timer() - start) / 60},
                 step=log_step)
            
            #if log_video:
            #    wandb.log({"Actor": utils.capture_video(actor, env_fn(), device=device)}, step=log_step)
            
            params = {f"actor-critic/param/{name}" : wandb.Histogram(param.detach().cpu()) for name, param in actor_critic.named_parameters()}
            grads = {f"actor-critic/gradient/{name}" : wandb.Histogram(param.grad.cpu()) for name, param in actor_critic.named_parameters()}
            wandb.log(params, step=log_step)
            wandb.log(grads, step=log_step)
            
            # Clear old logging data but keep the unfinished episode
            episode_lengths = [0]
            episode_returns = [0]
            
            # Advance log_step
            log_step += 1

class SequentialLstm(nn.Module):
    def __init__(self, inp_layers, lstm_size, out_layers, learn_h=False, learn_c=False):
        super().__init__()
        
        self.inp_net = utils.create_feedforward(inp_layers)
        self.out_net = utils.create_feedforward(out_layers)
        
        self.hx, self.hc = (None, None)
        self.initial_h = torch.zeros((1, lstm_size,), requires_grad=learn_h)
        self.initial_c = torch.zeros((1, lstm_size,), requires_grad=learn_c)
        self.lstm = nn.LSTMCell(inp_layers[-1], lstm_size)
    
    def forward(self, X):
        if self.hx is None:
            self.hx = self.initial_h
            self.hc = self.initial_c
        
        out = self.inp_net(X)
        (self.hx, self.cx) = self.lstm(out, (self.hx, self.hc))
        out = self.out_net(self.hx)
        return out
    
    def reset(self):
        self.hx = self.initial_h
        self.cx = self.initial_c
    
    @classmethod
    def from_config(cls, json_config):
        return cls(
            json_config["input_layers"],
            json_config["lstm_size"],
            json_config["output_layers"],
            learn_h=json_config.get("learn_h", False),
            learn_c=json_config.get("learn_c", False))
        
class PtbRecurrentActorCritic(nn.Module):
    def __init__(self, embedding_size, actor: nn.Module, critic: nn.Module):
        super().__init__()
        
        self.embedding = nn.Embedding(256, embedding_size)
        self.actor = actor
        self.critic = critic
        
    def forward(self, X):
        embedded_X = self.embedding(X.long()).reshape(X.shape[0], -1)
        actor_logits = self.actor(embedded_X)
        values = self.critic(embedded_X)
        return torch.distributions.Categorical(logits=actor_logits), values
    
    @torch.no_grad()
    def get_values(self, X):
        embedded_X = self.embedding(X.long()).reshape(X.shape[0], -1)
        values = self.critic(embedded_X)
        return values.cpu().numpy()
    
    @torch.no_grad()
    def get_actions(self, X):
        embedded_X = self.embedding(X.long()).reshape(X.shape[0], -1)
        actor_logits = self.actor(embedded_X)
        dist = torch.distributions.Categorical(logits=actor_logits)
        return dist.sample().cpu().numpy()
    
    def reset(self):
        self.actor.reset()
        self.critic.reset()
    
    @classmethod
    def from_config(cls, json_config):
        actor = SequentialLstm.from_config(json_config["actor"])
        critic = SequentialLstm.from_config(json_config["critic"])
        return cls(json_config["embedding_size"], actor, critic)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config.json")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as config:
            config = json.load(config)
    except:
        print(f"Error could not parse '{args.config}'.")
        from sys import exit
        exit(0)
    
    # Set seed for deterministic results
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Initialize actor-critic
    actor_critic = PtbRecurrentActorCritic.from_config(config["actor_critic"])
    
    # Create environment function
    env_fn = lambda: gym.make('PressTheButton-v0')
    
    # Setup Logging
    full_experiment_name = f"{config['experiment_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_folder = os.path.join(config["log_folder"], full_experiment_name)
    print(f"Log Folder '{log_folder}'")
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    wandb.init(name=full_experiment_name, project='RL-Algorithms', dir=log_folder)
    
    # Train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    actor_critic.to(device)
    
    print("---- STARTING TRAINING ----")
    train(actor_critic, env_fn,
          train_steps=config["train_steps"], env_steps=config["batch_size"],
          reward_decay=config["reward_decay"], device=device,
          log_folder=log_folder, save_frequency=config["log_frequency"])
