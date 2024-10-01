# mcts.py
import random
import math
import torch
from gym_env import Gym2048Env
from game import Board

# Maximum depth of the MCTS tree
MAX_DEPTH = 50

# Maximum number of moves in a single simulation
MAX_SIM_MOVES = 5

# Number of simulation runs for each leaf node evaluation
MAX_SIM_RUNS = 5

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    """
    def __init__(self, move, board, parent=None):
        self.move = move
        self.visit_count = 0
        self.value_sum = 0
        self.parent = parent
        self.board = board
        self.children = {}
        self.prior = 0
        self.depth = 0 if parent is None else parent.depth + 1

class MCTS:
    """
    Monte Carlo Tree Search implementation for 2048 game.
    """
    def __init__(self, value_net, policy_net, num_simulations, c_puct):
        self.value_net = value_net
        self.policy_net = policy_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def search(self, env):
        """
        Perform MCTS search to find the best move.
        """
        root = MCTSNode(None, Board(otherboard=env.board))

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.children and not node.board.is_game_over() and node.depth < MAX_DEPTH:
                action, node = self.select_child(node)
                search_path.append(node)

            # Expansion and evaluation
            if not node.board.is_game_over():
                state = node.board.board_array
                state_tensor = torch.FloatTensor(env.one_hot_encode()).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    policy, value = self.policy_net(state_tensor), self.value_net(state_tensor)
                policy = policy.squeeze().cpu().numpy()
                value = value.item()

                self.expand(node, policy)
                
                # Simulation
                simulation_value = self.simulation(node.board)
                value = (value + simulation_value) / 2  # Combine network value and simulation value
                value = math.tanh(value)
            else:
                value = -1  # Game over or max depth reached

            self.backpropagate(search_path, value)

        # Select action (most visited child)
        return self.select_most_visited_child(root)

    def select_child(self, node):
        """
        Select the child node with the highest UCB score.
        """
        max_ucb = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            ucb = self.get_ucb(node, child)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def get_ucb(self, parent, child):
        """
        Calculate the UCB (Upper Confidence Bound) score for a child node.
        """
        if child.visit_count == 0:
            return float('inf')
        
        q_value = child.value_sum / child.visit_count
        u_value = self.c_puct * child.prior * math.sqrt(math.log(parent.visit_count) / child.visit_count)
        return q_value + u_value

    def expand(self, node, policy):
        """
        Expand the current node by adding child nodes for each possible move.
        """
        for action, direction in enumerate(["up", "down", "left", "right"]):
            new_board = Board(otherboard=node.board)
            if new_board.move(direction):
                child = MCTSNode(action, new_board, parent=node)
                child.prior = policy[action]
                node.children[action] = child

    def simulation(self, board, num_moves=MAX_SIM_MOVES, num_runs=MAX_SIM_RUNS):
        """
        Perform random simulations from the current board state using the Gym environment.
        """
        total_score = 0
        for _ in range(num_runs):
            run_score = 0
            moves = 0
            env = Gym2048Env()
            env.load_board(board)
            done = False
            while moves < num_moves and not done:
                action = env.action_space.sample() 
                _, reward, done, _ = env.step(action)
                run_score += reward
                moves += 1
            total_score += run_score / (moves + 1)
        return total_score / num_runs

    def backpropagate(self, search_path, value):
        """
        Backpropagate the evaluation results through the search path.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

    def select_most_visited_child(self, node):
        """
        Select the child node with the highest visit count.
        """
        return max(node.children.items(), key=lambda x: x[1].visit_count)[0]

def play_game(mcts, env):
    """
    Play a single game using the MCTS algorithm.
    """
    env.reset()
    done = False
    total_reward = 0
    while not done:
        move = mcts.search(env)
        _, reward, done, info = env.step(move)
        total_reward += reward
        print(env.render())
        print(f"Action: {['up', 'down', 'left', 'right'][move]}, Reward: {reward}, Done: {done}")
    print(f"Game over. Max tile: {info['max_tile']}, Score: {info['score']}")
    return total_reward, info['max_tile'], info['score']

class DummyValueNetwork(torch.nn.Module):
    """
    A dummy value network that returns random values.
    """
    def __init__(self):
        super(DummyValueNetwork, self).__init__()

    def forward(self, x):
        return torch.rand(1)

class DummyPolicyNetwork(torch.nn.Module):
    """
    A dummy policy network that returns random probabilities for actions.
    """
    def __init__(self):
        super(DummyPolicyNetwork, self).__init__()

    def forward(self, x):
        policy = torch.rand(4)
        return policy / policy.sum()

def main():
    """
    Main function to run the MCTS algorithm on the 2048 game.
    """
    env = Gym2048Env()
    value_net = DummyValueNetwork()
    policy_net = DummyPolicyNetwork()
    mcts = MCTS(value_net, policy_net, num_simulations=100, c_puct=2)
    
    num_games = 1
    for i in range(num_games):
        total_reward, max_tile, score = play_game(mcts, env)
        print(f"Game {i+1}: Total Reward: {total_reward}, Max Tile: {max_tile}, Score: {score}")

if __name__ == "__main__":
    main()
