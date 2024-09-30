# azero.py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from gym_env import Gym2048Env
from models import ValueNetwork, PolicyNetwork
from mcts import MCTS
import os
import matplotlib.pyplot as plt
import json


class AlphaZero:
    def __init__(self, env, value_net, policy_net, mcts, learning_rate, num_episodes, save_path='alpha_zero_model.pth'):
        self.env = env
        self.value_net = value_net
        self.policy_net = policy_net
        self.mcts = mcts
        self.optimizer = optim.Adam(list(value_net.parameters()) + list(policy_net.parameters()), lr=learning_rate)
        self.num_episodes = num_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_net.to(self.device)
        self.policy_net.to(self.device)
        self.save_path = save_path
        self.start_episode = 0
        self.total_rewards = []
        self.avg_rewards = []
        self.losses = []
        self.avg_losses = []

    def save_model(self, episode):
        torch.save({
            'episode': episode,
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_rewards': self.total_rewards,
            'losses': self.losses,
            'reward_weights': self.env.weights
        }, self.save_path)

    def load_model(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.start_episode = checkpoint['episode'] + 1  # Start from the next episode
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_rewards = checkpoint['total_rewards']
            self.losses = checkpoint['losses']
            print(f"Resuming training from episode {self.start_episode}")
            return True
        return False

    def update_graph(self, episode, hyperparams):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.avg_rewards)
        plt.title(f"Reward and Loss over Episodes\n{hyperparams}", fontsize=10, pad=20)
        plt.ylabel("Average Reward")
        
        plt.subplot(2, 1, 2)
        plt.plot(self.avg_losses)
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        
        plt.tight_layout()
        plt.savefig('reward_loss_graph.png')
        plt.close()

    def train(self, hyperparams):
        if self.load_model():
            episode_range = range(self.start_episode, self.num_episodes)
        else:
            episode_range = range(self.num_episodes)
            
        for episode in episode_range:
            reward = self.play_game()
            self.total_rewards.append(reward)
            self.update_networks()
            
            if (episode + 1) % 1 == 0:
                avg_reward = sum(self.total_rewards[-10:]) / 10
                avg_loss = sum(self.losses[-10:]) / 10 if self.losses else 0
                self.avg_rewards.append(avg_reward)
                self.avg_losses.append(avg_loss)
                print(f"Completed episode {episode + 1}, Average reward: {avg_reward:.2f}, Average loss: {avg_loss:.4f}")
                self.update_graph(episode + 1, hyperparams)
            
            if (episode + 1) % 50 == 0:
                print(f"Sample game at episode {episode + 1}:")
                self.play(num_games=1, display=True)
            
            # Save the model after each episode
            self.save_model(episode)

    def play_game(self):
        state = self.env.reset()
        done = False
        self.game_history = []
        total_reward = 0

        while not done:
            action = self.mcts.search(self.env)
            next_state, reward, done, _ = self.env.step(action)
            self.game_history.append((state, action, reward))
            state = next_state
            total_reward += reward

        return total_reward

    def update_networks(self):
        total_loss = 0
        batch_size = 64
        num_batches = (len(self.game_history) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.game_history))
            batch = self.game_history[start_idx:end_idx]

            states, actions, rewards = zip(*batch)
            state_tensors = torch.stack([torch.FloatTensor(self.env.one_hot_encode(state)) for state in states]).to(self.device)
            action_tensors = torch.LongTensor(actions).to(self.device)
            reward_tensors = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)

            value_preds = self.value_net(state_tensors)
            policy_preds = self.policy_net(state_tensors)

            value_loss = F.mse_loss(value_preds, reward_tensors)
            policy_loss = F.cross_entropy(policy_preds, action_tensors)

            loss = value_loss + policy_loss
            total_loss += loss.item() * len(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.game_history)
        self.losses.append(avg_loss)

    def play(self, num_games, display=False):
        for game in range(num_games):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.mcts.search(self.env)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if display:
                    print(f"Action: {action}, Reward: {reward}")
                    print(self.env.board)

            print(f"Game {game + 1} over. Total reward: {total_reward}")
        return total_reward / num_games

    def sample_network_performance(self, num_games):
        total_rewards = []
        for game in range(num_games):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(self.env.one_hot_encode()).unsqueeze(0).to(self.device)
                policy_output = self.policy_net(state_tensor).squeeze()
                print(policy_output)
                action = torch.argmax(policy_output).item()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                print(self.env.render())

            total_rewards.append(total_reward)
            print(f"Game {game + 1} over. Total reward: {total_reward}")

        avg_reward = sum(total_rewards) / num_games
        print(f"Average reward over {num_games} games: {avg_reward:.2f}")
        return avg_reward

if __name__ == "__main__":
    # Hyperparameters
    input_size = 256
    hidden_size = 512
    num_simulations = 50 
    c_puct = 2 
    learning_rate = 0.000001
    num_episodes = 20
    num_games_to_play = 1
    num_games_to_sample = 1

    # Create a string of hyperparameters for the graph
    hyperparams = f"Input Size: {input_size}, Hidden Size: {hidden_size}\n" \
                  f"Num Simulations: {num_simulations}, C_puct: {c_puct}\n" \
                  f"Learning Rate: {learning_rate}, Num Episodes: {num_episodes}"

    # Initialize environment and networks
    env = Gym2048Env()
    value_net = ValueNetwork(input_size, hidden_size)
    policy_net = PolicyNetwork(input_size, hidden_size)
    mcts = MCTS(value_net, policy_net, num_simulations, c_puct)
    alpha_zero = AlphaZero(env, value_net, policy_net, mcts, learning_rate, num_episodes)

    summary(policy_net, (1,input_size))
    summary(value_net, (1,input_size))
    # Train the agent
    print("Starting training...")
    alpha_zero.train(hyperparams)
    print("Training completed.")
    # Sample network performance with MCTS
    alpha_zero.play(num_games=num_games_to_sample, display=True)
    # Sample network performance without MCTS
    # print(f"Sampling network performance over {num_games_to_sample} games without MCTS...")
    # alpha_zero.sample_network_performance(num_games_to_sample)
