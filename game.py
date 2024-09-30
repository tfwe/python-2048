import numpy as np
import random

class Board:
    def __init__(self, size=4, otherboard=None):
        if otherboard is None:
            self.size = max(size, 2)
            self.game_over = False
            self.turn = 0
            self.prev_move = None
            self.prev_board = np.zeros((self.size, self.size), dtype=int)
            self.board_array = np.zeros((self.size, self.size), dtype=int)
            self.score = 0
        else:
            self.size = otherboard.size
            self.game_over = otherboard.game_over
            self.turn = otherboard.turn
            self.prev_move = otherboard.prev_move
            self.prev_board = np.copy(otherboard.prev_board)
            self.board_array = np.copy(otherboard.board_array)
            self.score = otherboard.score

    def __str__(self):
        return f"2048 (Turn: {self.turn}, Score: {self.score}, Game over: {self.game_over})\n{self.board_string()}"

    def board_string(self):
        return '\n'.join([' '.join([f"{tile:4d}" for tile in row]) for row in self.board_array])

    def reset(self):
        self.__init__(self.size)
        self.start_game()

    def start_game(self):
        self.spawn_pieces()
        self.spawn_pieces()
        self.game_over = False
        self.prev_move = None
        self.turn = 0
        self.score = 0

    def spawn_pieces(self):
        open_tiles = self.get_open_tiles()
        for _ in range(min(len(open_tiles), 1)):
            y, x = open_tiles.pop(np.random.randint(len(open_tiles)))
            self.board_array[y, x] = 2 if np.random.random() < 0.9 else 4

    def move(self, direction):
        self.prev_board = np.copy(self.board_array)
        prev_score = self.score

        if direction == "up":
            self._move_up()
        elif direction == "down":
            self._move_down()
        elif direction == "left":
            self._move_left()
        elif direction == "right":
            self._move_right()

        if not np.array_equal(self.prev_board, self.board_array):
            self.spawn_pieces()
            self.turn += 1
            self.prev_move = direction
            return True
        return False

    def _move_up(self):
        self._move_and_merge(range(1, self.size), range(self.size), -1, 0)

    def _move_down(self):
        self._move_and_merge(range(self.size - 2, -1, -1), range(self.size), 1, 0)

    def _move_left(self):
        self._move_and_merge(range(1, self.size), range(self.size), 0, -1)

    def _move_right(self):
        self._move_and_merge(range(self.size - 2, -1, -1), range(self.size), 0, 1)

    def _move_and_merge(self, row_range, col_range, row_step, col_step):
        for row in row_range:
            for col in col_range:
                if self.board_array[row, col] != 0:
                    self._move_tile(row, col, row_step, col_step)

    def _move_tile(self, row, col, row_step, col_step):
        current_row, current_col = row, col
        while 0 <= current_row + row_step < self.size and 0 <= current_col + col_step < self.size:
            next_row, next_col = current_row + row_step, current_col + col_step
            if self.board_array[next_row, next_col] == 0:
                self.board_array[next_row, next_col] = self.board_array[current_row, current_col]
                self.board_array[current_row, current_col] = 0
                current_row, current_col = next_row, next_col
            elif self.board_array[next_row, next_col] == self.board_array[current_row, current_col]:
                self.board_array[next_row, next_col] *= 2
                self.score += self.board_array[next_row, next_col]
                self.board_array[current_row, current_col] = 0
                break
            else:
                break

    def get_open_tiles(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board_array[i, j] == 0]

    def get_available_moves(self):
        moves = []
        for direction in ["up", "down", "left", "right"]:
            test_board = Board(otherboard=self)
            if test_board.move(direction):
                moves.append(direction)
        return moves

    def make_random_move(self):
        """
        Make a random valid move on the board.
        
        Returns:
        - str: The direction of the move made, or None if no valid moves are available.
        """
        available_moves = self.get_available_moves()
        if not available_moves:
            self.game_over = True
            return None
        
        random_move = random.choice(available_moves)
        self.move(random_move)
        return random_move

    def is_game_over(self):
        self.game_over = len(self.get_available_moves()) == 0
        return self.game_over
