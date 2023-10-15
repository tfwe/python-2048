import numpy as np
import random
import math

class Board(object):
    def __init__(self, size=4, otherboard=None):
        if otherboard is None:
            if size < 2:
                size = 2
            self.size = size
            self.game_over = False
            self.turn = 0
            self.board_array = np.zeros((size,size)).astype(int)
            self.initial_array = np.zeros((size,size)).astype(int)
        else:
            self.size = otherboard.size
            self.game_over = otherboard.game_over
            self.turn = otherboard.turn
            self.board_array = np.copy(otherboard.board_array)
            self.initial_array = np.copy(otherboard.initial_array)
    
    def __str__(self):
        return f"2048 (Turn: {self.turn}, Game over: {self.game_over})\n{self.board_array}"

    def start_game(self):
        self.spawn_pieces()
        self.spawn_pieces()

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

    def move(self, direction):
        possible_moves = {
                "up":self.move_up,
                "down":self.move_down,
                "right":self.move_right,
                "left":self.move_left
                }
        if not possible_moves[direction]:
            return False
        return possible_moves[direction]()

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
                    elif self.board_array[y - 1][x] == self.board_array[y][x] and merge_allowed[x]:
                        self.board_array[y - 1][x] *= 2
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
        sim_board = Board(-1, self)
        for i in possible_moves:
            if sim_board.move(i):
                available_moves.append(i)
                sim_board = Board(-1, self)
        if len(available_moves) == 0:
            self.game_over = True
        return available_moves
    
    def random_move(self):
        moves = self.get_available_moves()
        if len(moves) == 0:
            return False
        rand_move = lambda moves: self.move(moves[0 if len(moves) == 1 else random.randint(0, len(moves) - 1)])
        return rand_move(moves)

def main():
    board = Board(4)
    board.start_game()
    print(board)
    moves = board.get_available_moves() 
    print(moves)
    while len(moves) >= 1:
        board.random_move()
        moves = board.get_available_moves() 
        print(board)

if __name__ == "__main__":
    main()
