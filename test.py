# test.py
import math
import numpy as np
import torch
from gym_env import Gym2048Env
from models import ValueNetwork, PolicyNetwork
from mcts import MCTS
from azero import AlphaZero
import json
from scipy.optimize import minimize

def save_weights(weights, weights_file):
    """
    Save weights to a file and print them to stdout.
    """
    with open(weights_file, 'w') as f:
        json.dump(weights, f)
    print(f"Current weights: {weights}")

def select_parameters_to_optimize():
    """
    Allow the user to select which weights and parameters to optimize.
    """
    all_parameters = ['max_tile', 'merge', 'turn', 'new_max', 'monotonicity', 'corner', 'scale', 'c_puct']
    print("Available parameters to optimize:", all_parameters)
    selected = input("Enter the parameters you want to optimize (comma-separated): ").split(',')
    return [p.strip() for p in selected if p.strip() in all_parameters]

def objective_function(params_to_optimize, fixed_params, selected_params, env, value_net, policy_net, num_simulations, num_games, weights_file):
    """
    Objective function to minimize. It runs games with given weights and c_puct, and returns the negative of the average score.
    """
    # Combine fixed and optimized parameters
    all_params = fixed_params.copy()
    for i, param in enumerate(selected_params):
        all_params[param] = params_to_optimize[i]

    # Extract c_puct and weights
    c_puct = all_params.pop('c_puct')
    weights = all_params

    # Scale weights to [0, 1]
    max_weight = max(weights.values())
    scaled_weights = {k: v / max_weight for k, v in weights.items()}
    
    env.weights = scaled_weights
    save_weights({**scaled_weights, 'c_puct': c_puct}, weights_file)  # Save and print weights and c_puct
    
    mcts = MCTS(value_net, policy_net, num_simulations, c_puct)
    alpha_zero = AlphaZero(env, value_net, policy_net, mcts, 0.000000001, 1)  # Learning rate and num_episodes not used here
    
    total_reward = alpha_zero.play(num_games=num_games, display=False)
    
    avg_reward = math.log(total_reward / num_games + 1)
    print(f"Average reward: {avg_reward}, c_puct: {c_puct}")
    return -avg_reward  # Negative because we want to maximize the score

def main():
    # Hyperparameters
    input_size = 256
    hidden_size = 512
    num_simulations = 1 
    learning_rate = 0.000000001
    num_episodes = 20
    num_games_to_play = 1
    num_games_to_sample = 100  # Number of games to play for each optimization step
    weights_file = 'weights.json'  # Path to the weights file

    # Initialize environment and networks
    env = Gym2048Env(weights_file=weights_file)
    value_net = ValueNetwork(input_size, hidden_size)
    policy_net = PolicyNetwork(input_size, hidden_size)

    # Initial weights and c_puct from file
    with open(weights_file, 'r') as f:
        initial_params = json.load(f)
    
    # Ensure c_puct is in the initial parameters
    if 'c_puct' not in initial_params:
        initial_params['c_puct'] = 0.2  # Default value if not present

    # Print initial parameters
    print("Initial parameters:")
    save_weights(initial_params, weights_file)

    # Select parameters to optimize
    selected_params = select_parameters_to_optimize()
    
    if not selected_params:
        print("No parameters selected for optimization. Exiting.")
        return

    # Prepare initial values and bounds for selected parameters
    initial_values_to_optimize = [initial_params[p] for p in selected_params]
    bounds = [(0, 1) if p != 'c_puct' else (0.01, 2) for p in selected_params]

    # Use SciPy's minimize function to find optimal parameters
    result = minimize(
        objective_function,
        initial_values_to_optimize,
        args=(initial_params, selected_params, env, value_net, policy_net, num_simulations, num_games_to_sample, weights_file),
        method='L-BFGS-B',
        bounds=bounds
    )

    optimal_values = result.x
    print(f"Optimization completed. Final parameters:")

    # Update initial_params with optimized values
    for i, param in enumerate(selected_params):
        initial_params[param] = optimal_values[i]

    # Save and print optimal parameters
    save_weights(initial_params, weights_file)

    # Update environment with new weights
    env.weights = {k: v for k, v in initial_params.items() if k != 'c_puct'}

    # Initialize MCTS and AlphaZero with optimal weights and c_puct
    mcts = MCTS(value_net, policy_net, num_simulations, initial_params['c_puct'])
    alpha_zero = AlphaZero(env, value_net, policy_net, mcts, learning_rate, num_episodes)

    # Play some games with the optimized agent using MCTS
    print(f"Playing {num_games_to_play} games with the optimized agent using MCTS...")
    alpha_zero.play(num_games=num_games_to_play, display=True)

if __name__ == "__main__":
    main()
