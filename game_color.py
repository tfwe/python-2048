import numpy as np
import random
import math
from colorama import Fore, Style, Back
import termplotlib as tpl 
plt = tpl.figure()
class Board(object):
    def __init__(self, size=4, otherboard=None):
        if otherboard is None:
            if size < 2:
                size = 2
            self.size = size
            self.game_over = False
            self.turn = 0
            self.prev_move = None
            self.prev_board = np.zeros((size,size)).astype(int)
            self.board_array = np.zeros((size,size)).astype(int)
            self.initial_array = np.zeros((size,size)).astype(int)
        else:
            self.size = otherboard.size
            self.game_over = otherboard.game_over
            self.turn = otherboard.turn
            self.prev_move = otherboard.prev_move
            self.prev_board = otherboard.prev_board
            self.board_array = np.copy(otherboard.board_array)
            self.initial_array = np.copy(otherboard.initial_array)
    
    def __str__(self):
        return f"2048 (Turn: {self.turn}, Game over: {self.game_over})\n{self.board_string()}"
    
    def board_string(self):
        board_str = ""
        
        def prettify_tile(value, length=4):
            if value == 0:
                return f'{Style.RESET_ALL}{Back.LIGHTWHITE_EX}{str("").center(length)}'
                # return f'{Back.LIGHTBLACK_EX} {str(value).ljust(4)}'
            elif value == 2:
                return f'{Fore.LIGHTBLACK_EX}{Back.WHITE}{str(value).center(length)}' 
            elif value == 4:
                return f'{Fore.LIGHTBLACK_EX}{Back.WHITE}{str(value).center(length)}' 
            elif value == 8:
                return f'{Style.RESET_ALL}{Back.LIGHTRED_EX}{str(value).center(length)}' 
            elif value == 16:
                return f'{Style.RESET_ALL}{Back.LIGHTRED_EX}{str(value).center(length)}' 
            elif value == 32:
                return f'{Style.RESET_ALL}{Back.RED}{str(value).center(length)}'
            elif value == 64:
                return f'{Style.RESET_ALL}{Back.RED}{str(value).center(length)}'
            elif value == 128:
                return f'{Style.RESET_ALL}{Back.YELLOW}{str(value).center(length)}'
            elif value == 256:
                return f'{Style.RESET_ALL}{Back.YELLOW}{str(value).center(length)}'
            elif value == 512:
                return f'{Style.RESET_ALL}{Back.YELLOW}{str(value).center(length)}'
            elif value == 1024:
                return f'{Style.RESET_ALL}{Back.YELLOW}{str(value).center(length)}'
            elif value == 2048:
                return f'{Style.RESET_ALL}{Fore.LIGHTBLACK_EX}{Back.LIGHTYELLOW_EX}{str(value).center(length)}'
            else:
                return f'{Style.RESET_ALL}{Back.BLACK}{str(value).center(length)}'
        
        for i in range(self.size):
            for j in range(self.size):
                board_str += Style.RESET_ALL
                board_str += prettify_tile(self.board_array[i][j]) 
                board_str += Style.RESET_ALL 
            board_str += '\n'
        
        return board_str    
    def reset(self):
        self = Board(self.size)
        self.start_game()

    def start_game(self):
        self.initial_array = np.copy(self.board_array)
        self.prev_board = np.copy(self.board_array)
        self.spawn_pieces()
        self.spawn_pieces()
        self.game_over = False
        self.prev_move = None
        self.turn = 0

    def load_array(self, board_array):
        self.board_array = np.copy(board_array)
        self.size = board_array.shape[0]
        self.turn = 0
        available_moves = self.get_available_moves() # sets game over flag

    def spawn_pieces(self):
        possible_starting_pieces = {2: 0.9, 4: 0.1}  # Modify this dictionary to add more tiles in the future
        open_tiles = self.get_open_tiles()

        for _ in range(math.ceil(self.size / 4)):
            if len(open_tiles) == 0:
                return
            elif len(open_tiles) == 1:
                rand_tile_index = 0
            else:
                rand_tile_index = random.randint(0, len(open_tiles) - 1)
                random.shuffle(open_tiles)
            rand_piece = random.choices(list(possible_starting_pieces.keys()), weights=possible_starting_pieces.values(), k=1)[0]
            y = open_tiles[rand_tile_index][0]
            x = open_tiles[rand_tile_index][1]
            self.board_array[y][x] = rand_piece
        
    def move(self, direction=None, index=0):
        possible_moves = {
            "up":self.move_up,
            "down":self.move_down,
            "right":self.move_right,
            "left":self.move_left
            }
        choices = ["up","down","left","right"]
        if direction == None:
            direction = possible_moves[choices[index]]
        result = possible_moves[direction]()
        if not result:
            return False
        self.prev_board = np.copy(self.board_array)
        self.prev_move = direction
        return result

    def move_up(self):
        x = 0
        y = 0
        board_changed = False
        initial_board = self.board_array
        for i in range(self.size):
            merge_allowed = [True]*self.size
            for j in range(1, self.size):
                x = i
                y = j
                if self.board_array[y][x] == 0:
                    continue
                while y > 0:
                    if self.board_array[y - 1][x] == 0:
                        self.board_array[y - 1][x] = self.board_array[y][x]
                        self.board_array[y][x] = 0
                        y -= 1
                        board_changed = True
                    elif math.fabs(self.board_array[y - 1][x]) == math.fabs(self.board_array[y][x]) and merge_allowed[x]:
                        self.board_array[y - 1][x] += self.board_array[y][x]
                        self.board_array[y][x] = 0
                        merge_allowed[x] = False
                        y -= 1
                        board_changed = True
                    else:
                        break
        if board_changed:
            self.spawn_pieces()
            self.turn += 1
            return True
        return False

    def move_down(self):
        self.board_array = np.rot90(self.board_array, 2, (1,0))
        is_successful = self.move_up()
        self.board_array = np.rot90(self.board_array, 2, (1,0))
        return is_successful

    def move_left(self):
        self.board_array = np.rot90(self.board_array, 1, (1,0))
        is_successful = self.move_up()
        self.board_array = np.rot90(self.board_array, 1, (0,1))
        return is_successful

    def move_right(self):
        self.board_array = np.rot90(self.board_array, 1, (0,1))
        is_successful = self.move_up()
        self.board_array = np.rot90(self.board_array, 1, (1,0))
        return is_successful
   
    def get_open_tiles(self):
        open_tiles = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board_array[i][j] == 0:
                    open_tiles.append((i,j))
        return open_tiles

    def get_available_moves(self):
        available_moves = []
        possible_moves = ["up", "down", "right", "left"]
        sim_board = Board(otherboard=self)
        for i in possible_moves:
            if sim_board.move(i):
                available_moves.append(i)
                sim_board = Board(otherboard=self)
                sim_board = Board(-1, self)
        return np.array(available_moves)
   
    def is_game_over(self):
        available_moves = self.get_available_moves()
        if len(available_moves) == 0:
            self.game_over = True
        return self.game_over

    def random_move(self):
        moves = self.get_available_moves()
        if not self.is_game_over():
            if len(moves) == 0:
                return False
            rand_move = lambda moves: self.move(moves[0 if len(moves) == 1 else random.randint(0, len(moves) - 1)])
            return rand_move(moves)
        return False
    
    def random_sample_game(self, print_board=False, size=4):
        board = Board(size)
        board.start_game()
        tiles = {}
        while not board.is_game_over():
            if print_board:
                print(board)
            board.random_move()
        print(board)
            # board.get_new_tiles()

        max_tile = np.max(board.board_array)
        num_turns = board.turn
        return num_turns,max_tile
    

def main():
    b = Board()

    scores_random, best_tiles_random = [], []
    for i in range(100):
        if i % 100 == 0:
            print(f"Iteration {i}")
        total_score, best_tile = b.random_sample_game()
        scores_random.append(total_score)
        best_tiles_random.append(best_tile)
    print("Finish")

    plt.hist(counts=np.histogram(scores_random, bins = 50)[0], bin_edges=np.histogram(scores_random, bins = 50)[1], orientation="horizontal")
    # plt.title("Total score distribution")
    # plt.xlabel("Total Score")
    # plt.ylabel("Frequency")
    plt.show()
    #
    # max_power = int(math.log(max(best_tiles_random), 2)) + 1
    # min_power = int(math.log(min(best_tiles_random), 2))
    # unique, counts = np.unique(best_tiles_random, return_counts=True)
    # plt.barh([str(2 ** i) for i in range(min_power, max_power)], counts)
    # plt.title("Best tile distribution")
    # plt.xlabel("Best tile")
    # plt.ylabel("Frequency")
    # plt.show()
if __name__ == "__main__":
    main()

