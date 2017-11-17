import time
import random

class rotated_piece(object):
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid[0])
        self.width = len(grid)

    def __str__(self):
        return str(self.grid)
    def __repr__(self):
        return "rotated_piece(" + repr(self.grid) + ")"

class piece(object):
    def __init__(self, grid, num_rotations):
        self.grid = grid
        self.num_rotations = num_rotations

    def rotations(self):
        # Generator that returns all the rotations for this piece.
        rotated_grid = self.grid
        for i in range(self.num_rotations):
            yield rotated_piece(rotated_grid)
            rotated_grid = list(zip(*rotated_grid[::-1]))

def pieces():
    # generator that returns all the non-rotated pieces
    yield piece(((1,),
                 (1,),
                 (1,),
                 (1,)), 2)

    yield piece(((0, 1, 0),
                 (1, 1, 1)), 4)

    yield piece(((1, 0),
                 (1, 0),
                 (1, 1)), 4)

    yield piece(((0, 1),
                 (0, 1),
                 (1, 1)), 4)

    yield piece(((1, 1),
                 (1, 1)), 1)

    yield piece(((1, 0),
                 (1, 1),
                 (0, 1)), 2)

    yield piece(((0, 1),
                 (1, 1),
                 (1, 0)), 2)

class board(object):
    def __init__(self):
        self.board = self.make_board()

    def make_board(self):
        return [[0] * 20 for i in range(10)]

    def fits_row(self, rotated_piece, col):
        # Returns the one and only one row that the piece must be placed at in this column.
        if rotated_piece.width + col >= 10:
            return -1
        for offset_y in range(21):
            for piece_x in range(rotated_piece.width):
                for piece_y in range(rotated_piece.height):
                    if piece_y + offset_y == 20:
                        # Off the bottom of the board
                        return offset_y - 1;
                    if self.board[piece_x + col][piece_y + offset_y] \
                       + rotated_piece.grid[piece_x][piece_y] == 2:
                        # Did not fit
                        return offset_y-1

    def place(self, rotated_piece, x, y):
        for piece_x in range(rotated_piece.width):
            for piece_y in range(rotated_piece.height):
                if rotated_piece.grid[piece_x][piece_y] == 1:
                    self.board[piece_x + x][piece_y + y] = 1

        # Clear completed lines
        board2 = self.make_board()
        board2_row = 19
        for y in range(19, -1, -1):
            s = 0
            for x in range(10):
                s += self.board[x][y]
            if s != 10:
                for x in range(10):
                    board2[x][board2_row] = self.board[x][y]
                board2_row -= 1
        self.board = board2

    def drop(self, rotated_piece, col):
        row = self.fits_row(rotated_piece, col)
        self.place(rotated_piece, col, row)

    def is_game_over(self):
        pass

    def valid_moves(self, piece):
        # generator that returns all the valid moves for the given piece
        for rot in piece.rotations():
            for col in range(10):
                row = self.fits_row(rot, col)
                if row >= 0:
                    yield (rot, col, row)

    def __str__(self):
        ret = "+" + "-" * 10 + "+" + "\n"
        for y in range(20):
            ret += "|"
            for x in range(10):
                if self.board[x][y] == 0:
                    ret += " "
                else:
                    ret += "#"
            ret += "|\n"
        ret += "+" + "-" * 10 + "+" + "\n"
        return ret

def play_random_game():
    b = board()
    all_pieces = list(pieces())
    while(True):
        p = random.choice(all_pieces)
        moves = list(b.valid_moves(p))
        if len(moves) == 0:
            break
        (r, x, y) = random.choice(moves)
        b.place(r, x, y)
        print(b)
        time.sleep(.25)

if __name__ == "__main__":
    play_random_game()

