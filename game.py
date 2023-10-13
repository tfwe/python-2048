import numpy as np
import random
import math

class Board(object):
    def __init__(self, size=4):
        if size < 2:
            size = 2
        self.size = size
        self.game_over = False
        self.turn = 0
        self.board_array = np.zeros((size,size)).astype(int)
    
    def __str__(self):
        return f"2048 (Turn: {self.turn}, Game over: {self.game_over})\n{self.board_array}"

    def start_game(self):
        self.spawn_pieces()
        self.spawn_pieces()
    
    def spawn_pieces(self): 
        possible_starting_pieces = [2, 4]
        open_tiles = self.get_open_tiles()
        if len(open_tiles) == 0:
            return
        elif len(open_tiles) == 1:
            rand_piece_index = random.randint(0, len(possible_starting_pieces) - 1)
            self.board_array[open_tiles[0][0]][open_tiles[0][1]] = possible_starting_pieces[rand_piece_index]
            return
        else:
            random.shuffle(open_tiles)
            for k in range(math.ceil(self.size / 4)):
                rand_piece_index = random.randint(0, len(possible_starting_pieces) - 1)
                rand_tile_index = random.randint(0, len(open_tiles) - 1)
                y = open_tiles[rand_tile_index][0]
                x = open_tiles[rand_tile_index][1]
                self.board_array[y][x] = possible_starting_pieces[rand_piece_index]
                open_tiles.remove(open_tiles[rand_tile_index])

    def move(self, direction):
        if (direction == "up"):
            return self.move_up()
        elif (direction == "down"):
            return self.move_down()
        elif (direction == "right"):
            return self.move_right()
        elif (direction == "left"):
            return self.move_left()
        else:
            return False

    def move_up(self):
        x = 0
        y = 0
        board_changed = False
        initial_board = self.board_array
        for i in range(self.size):
            for j in range(self.size):
                x = j
                y = i
                if y == 0:
                    continue
                while y > 0:
                    if self.board_array[y][x] == 0:
                        pass
                    elif self.board_array[y - 1][x] == 0:
                        self.board_array[y - 1][x] = self.board_array[y][x]
                        self.board_array[y][x] = 0
                        board_changed = True
                    elif (self.board_array[y - 1][x] == self.board_array[y][x]):
                        self.board_array[y - 1][x] = self.board_array[y][x]*2
                        self.board_array[y][x] = 0
                        board_changed = True
                    else:
                        pass
                    y -= 1
        if board_changed:
            self.spawn_pieces()
            self.turn += 1
            if len(self.get_available_moves()) == 0:
                self.game_over = True
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
        for i in range(self.size):
            for j in range(self.size):
                if self.board_array[i][j] == 0:
                    # If there's an empty space, all moves are possible
                    return ["up", "down", "left", "right"]
                else:
                    # Check for same-value tiles horizontally
                    if j < self.size - 1 and self.board_array[i][j] == self.board_array[i][j + 1]:
                        if "left" not in available_moves:
                            available_moves.append("left")
                        if "right" not in available_moves:
                            available_moves.append("right")
                    # Check for same-value tiles vertically
                    if i < self.size - 1 and self.board_array[i][j] == self.board_array[i + 1][j]:
                        if "up" not in available_moves:
                            available_moves.append("up")
                        if "down" not in available_moves:
                            available_moves.append("down")
        return available_moves

def main():
    board = Board(7)
    board.start_game()
    moves = board.get_available_moves() 
    while len(moves) >= 1:
        if (len(moves) == 1):
            rand_index = 0
        else:
            rand_index = random.randint(0, len(moves) - 1)
        board.move(moves[rand_index])
        moves = board.get_available_moves()
        print(board)

if __name__ == "__main__":
    main()