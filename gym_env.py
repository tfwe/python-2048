import numpy as np
import gym
import math
from gym import spaces
from game import Board
import json

class Gym2048Env(gym.Env):
    """
    A custom Gym environment for the 2048 game.
    
    This environment allows an agent to play the 2048 game, with the goal of
    achieving the highest score possible by combining tiles with the same value.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, size=4, max_invalid_moves=5, weights_file='weights.json'):
        """
        Initialize the 2048 environment.

        Args:
            size (int): The size of the game board (default: 4).
            max_invalid_moves (int): Maximum number of consecutive invalid moves allowed (default: 5).
        """
        super(Gym2048Env, self).__init__()
        self.size = size
        self.board = Board(size)
        self.max_invalid_moves = max_invalid_moves
        self.consecutive_invalid_moves = 0
        
        # Define action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        
        # Define observation space to accommodate both encoding types
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(size, size), dtype=np.int32)
        
        # Load weights from file
        with open(weights_file, 'r') as f:
            self.weights = json.load(f)
        
        # Scale weights to [0, 1]
        max_weight = max(self.weights.values())
        self.weights = {k: v / max_weight for k, v in self.weights.items()}

    def encode(self, one_hot=False):
        """
        Encode the current game state based on the chosen encoding method.

        Args:
            one_hot (bool): Whether to use one-hot encoding (default: False).

        Returns:
            np.array: The encoded game state.
        """
        if one_hot:
            return self.one_hot_encode()
        else:
            return self.board.board_array.copy()

    def one_hot_encode(self, board_array=[]):
        """
        Perform one-hot encoding of the current game state.

        Returns:
            np.array: One-hot encoded game state.
        """
        if len(board_array) == 0:
            board_array = self.board.board_array
        encoded = np.zeros((16, self.size, self.size), dtype=np.float32)
        
        for i in range(self.size):
            for j in range(self.size):
                value = board_array[i, j]
                if value > 0:
                    power = int(np.floor(np.log2(value)))
                    if 0 <= power < 16:
                        encoded[power, i, j] = 1
        
        return encoded

    def calc_reward(self, valid_move, weights):
        """
        Calculate the reward for the current move, normalized between -1 and 1.

        Args:
            valid_move (bool): Whether the move was valid.
            weights (dict): Dictionary containing reward weights.

        Returns:
            float: The calculated reward, between -1 and 1.
        """
        if not valid_move:
            return -1.0  # Penalty for invalid moves

        # Calculate base rewards
        max_tile = np.max(self.board.board_array)
        open_tiles = math.tanh((np.count_nonzero(self.board.board_array == 0) - 5) / 2)

        # Reward for high max tile (encourages building large numbers)
        max_tile_reward = 1/(1 + math.exp(-(math.log2(max_tile)-4)/5))  # 11 is log2(2048), our target
        # print(max_tile_reward)
        # Reward for maintaining open tiles (encourages efficient board use)
        # open_tile_reward = open_tiles / 16  # 16 is the total number of tiles

        # Bonus for monotonicity (encourages ordered board)
        # monotonicity_bonus = self.calculate_monotonicity_bonus()

        # Bonus for reaching new max tiles
        # new_max_bonus = 0.5 if max_tile > np.max(self.board.prev_board) else 0.0

        # Penalty for not having max tile in corner
        corner_reward = self.calculate_corner_reward()

        # Bonus for merging tiles
        merge_bonus = self.calculate_merge_bonus()
        #
        # turn_bonus = math.log2(self.board.turn)
        #
        # Combine rewards using weights
        # total_reward = (
        #     weights['max_tile'] * max_tile_reward +
        #     weights['merge'] * merge_bonus +
        #     weights['turn'] * turn_bonus +
        #     weights['new_max'] * new_max_bonus + 
        #     weights['monotonicity'] * monotonicity_bonus -
        #     weights['corner'] * corner_penalty
        # )

        # Normalize to [-1, 1] range
        return np.mean([2*open_tiles, max_tile_reward, merge_bonus, corner_reward])

    def calculate_monotonicity_bonus(self):
        """
        Calculate a bonus for monotonicity in the board.
        Returns a value between 0 and 1.
        """
        board = self.board.board_array
        monotonicity_left = 0
        monotonicity_right = 0
        monotonicity_up = 0
        monotonicity_down = 0

        for i in range(4):
            for j in range(3):
                if board[i][j] != 0 and board[i][j+1] != 0:
                    if board[i][j] >= board[i][j+1]:
                        monotonicity_left += math.log2(board[i][j]) - math.log2(board[i][j+1])
                    else:
                        monotonicity_right += math.log2(board[i][j+1]) - math.log2(board[i][j])

                if board[j][i] != 0 and board[j+1][i] != 0:
                    if board[j][i] >= board[j+1][i]:
                        monotonicity_up += math.log2(board[j][i]) - math.log2(board[j+1][i])
                    else:
                        monotonicity_down += math.log2(board[j+1][i]) - math.log2(board[j][i])

        monotonicity = max(monotonicity_left, monotonicity_right) + max(monotonicity_up, monotonicity_down)
        return monotonicity / (11 * 4)  # Normalize by max possible monotonicity

    def calculate_corner_reward(self):
        """
        Calculate a penalty for not having the maximum tile in a corner.
        Returns a value between 0 and 1.
        """
        board = self.board.board_array
        max_tile = np.max(board)
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        
        if max_tile in corners:
            return 0  # No penalty if max tile is in a corner
        else:
            # Calculate the penalty based on the distance of max tile from the nearest corner
            max_tile_position = np.unravel_index(np.argmax(board), board.shape)
            distances = [
                abs(max_tile_position[0] - 0) + abs(max_tile_position[1] - 0),  # Top-left
                abs(max_tile_position[0] - 0) + abs(max_tile_position[1] - 3),  # Top-right
                abs(max_tile_position[0] - 3) + abs(max_tile_position[1] - 0),  # Bottom-left
                abs(max_tile_position[0] - 3) + abs(max_tile_position[1] - 3),  # Bottom-right
            ]
            min_distance = min(distances)
            return math.tanh((11-min_distance*math.log2(max_tile))/11)  # Normalize by max possible distance (3 + 3 = 6)

    def calculate_merge_bonus(self):
        """
        Calculate a bonus for merging tiles, scaled by the size of the resulting tile.
        Returns a value between 0 and 1.
        """
        current_board = self.board.board_array
        prev_board = self.board.prev_board
        
        # Find tiles that merged (tiles in current board that are twice the value of tiles in prev board)
        merged_tiles = []
        for i in range(4):
            for j in range(4):
                if current_board[i, j] != 0 and current_board[i, j] == 2 * prev_board[i, j]:
                    merged_tiles.append(current_board[i, j])
        
        if not merged_tiles:
            return 0  # No merges occurred
        
        # Calculate bonus based on the sum of log2 of merged tiles
        # This gives more weight to merges of larger tiles
        merge_bonus = sum(math.log2(tile) for tile in merged_tiles) / 11  # 11 is log2(2048)
        
        return min(merge_bonus, 1)  # Ensure the bonus is between 0 and 1

    def step(self, action):
        """
        Take a step in the environment by performing the given action.

        Args:
            action (int): The action to perform (0: up, 1: down, 2: left, 3: right).

        Returns:
            tuple: (observation, reward, done, info)
        """
        direction = ['up', 'down', 'left', 'right'][action]
        
        valid_move = self.board.move(direction)
        
        reward = self.calc_reward(valid_move, self.weights)
        
        done = self.board.is_game_over()
        if valid_move:
            self.consecutive_invalid_moves = 0
        else:
            self.consecutive_invalid_moves += 1
            if self.consecutive_invalid_moves >= self.max_invalid_moves:
                done = True
        
        info = {
            'max_tile': np.max(self.board.board_array),
            'score': np.sum(self.board.board_array),
            'valid_move': valid_move,
            'consecutive_invalid_moves': self.consecutive_invalid_moves,
        }
        return self.encode(), reward, done, info

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: The initial observation.
        """
        self.board.reset()
        self.board.start_game()
        self.consecutive_invalid_moves = 0
        return self.encode()

    def render(self, mode='human'):
        """
        Render the current state of the environment.

        Args:
            mode (str): The mode to render with (only 'human' is supported).

        Returns:
            str: A string representation of the game board.
        """
        return str(self.board)

    def close(self):
        """
        Clean up the environment. Currently does nothing.
        """
        pass

    def copy(self):
        """
        Create a deep copy of the current environment.

        Returns:
            Gym2048Env: A new instance of the environment with the same state.
        """
        new_env = Gym2048Env(size=self.size, max_invalid_moves=self.max_invalid_moves)
        new_env.board = Board(otherboard=self.board)
        new_env.consecutive_invalid_moves = self.consecutive_invalid_moves
        return new_env

    def load_board(self, board):
        """
        Load a given board state into the environment.

        Args:
            board (Board): The board state to load.
        """
        self.board = Board(otherboard=board)
        self.consecutive_invalid_moves = 0

if __name__ == "__main__":
    # Example usage of the environment
    env = Gym2048Env(weights_file='weights.json')
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        print(env.render())
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Action: {['up', 'down', 'left', 'right'][action]}, Reward: {reward}, Done: {done}")
        print(f"Consecutive Invalid Moves: {info['consecutive_invalid_moves']}")

    print(f"Game Over. Total Reward: {total_reward}, Max Tile: {info['max_tile']}")
