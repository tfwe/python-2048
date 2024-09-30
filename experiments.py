import numpy as np
import matplotlib.pyplot as plt
import torch
from gym_env import Gym2048Env
from dqn import DQNAgent, train

def run_experiment(hidden_size, learning_rate, gamma, buffer_size, batch_size, num_episodes):
    env = Gym2048Env()
    input_size = env.size ** 2 * 12  # 12 is the number of possible values (0 and 2^1 to 2^11)
    output_size = env.action_space.n
    
    agent = DQNAgent(input_size, output_size)
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)
    agent.gamma = gamma
    
    trained_agent = train(env, agent, num_episodes)
    
    # Play 10 games and get the average score
    scores = []
    for _ in range(10):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = trained_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        scores.append(total_reward)
    
    return np.mean(scores)

def visualize_hyperparameters():
    hidden_sizes = [64, 128, 256]
    learning_rates = [0.0001, 0.001, 0.01]
    gammas = [0.9, 0.99, 0.999]
    buffer_sizes = [5000, 10000, 20000]
    batch_sizes = [32, 64, 128]
    
    results = {}
    
    for param, values in [
        ("Hidden Size", hidden_sizes),
        ("Learning Rate", learning_rates),
        ("Gamma", gammas),
        ("Buffer Size", buffer_sizes),
        ("Batch Size", batch_sizes)
    ]:
        scores = []
        for value in values:
            if param == "Hidden Size":
                score = run_experiment(value, 0.001, 0.99, 10000, 64, 256)
            elif param == "Learning Rate":
                score = run_experiment(128, value, 0.99, 10000, 64, 256)
            elif param == "Gamma":
                score = run_experiment(128, 0.001, value, 10000, 64, 256)
            elif param == "Buffer Size":
                score = run_experiment(128, 0.001, 0.99, value, 64, 256)
            else:  # Batch Size
                score = run_experiment(128, 0.001, 0.99, 10000, value, 256)
            scores.append(score)
        results[param] = (values, scores)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Hyperparameter Optimization for 2048 RL Agent")
    
    for i, (param, (values, scores)) in enumerate(results.items()):
        ax = axs[i // 3, i % 3]
        ax.plot(values, scores, 'o-')
        ax.set_xlabel(param)
        ax.set_ylabel("Average Score")
        ax.set_title(f"Effect of {param}")
    
    plt.tight_layout()
    plt.savefig("hyperparameter_optimization.png")
    plt.close()

if __name__ == "__main__":
    visualize_hyperparameters()
