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
    yield piece((("I",),
                 ("I",),
                 ("I",),
                 ("I",)), 2)

    yield piece(((None, "T", None),
                 ("T", "T", "T")), 4)

    yield piece((("L", None),
                 ("L", None),
                 ("L", "L")), 4)

    yield piece(((None, "J"),
                 (None, "J"),
                 ("J", "J")), 4)

    yield piece((("O", "O"),
                 ("O", "O")), 1)

    yield piece((("S", None),
                 ("S", "S"),
                 (None, "S")), 2)

    yield piece(((None, "Z"),
                 ("Z", "Z"),
                 ("Z", None)), 2)

class board(object):
    def __init__(self):
        self.board = self.make_board()

    def make_board(self):
        return [[None] * 20 for i in range(10)]

    def fits_row(self, rotated_piece, col):
        # Returns the one and only one row that the piece must be placed at in this column.
        if rotated_piece.width + col > 10:
            return -1
        for offset_y in range(21):
            for piece_x in range(rotated_piece.width):
                for piece_y in range(rotated_piece.height):
                    if piece_y + offset_y == 20:
                        # Off the bottom of the board
                        return offset_y - 1;
                    if self.board[piece_x + col][piece_y + offset_y] != None and rotated_piece.grid[piece_x][piece_y] != None:
                        # Did not fit
                        return offset_y-1

    def place(self, rotated_piece, x, y):
        for piece_x in range(rotated_piece.width):
            for piece_y in range(rotated_piece.height):
                c = rotated_piece.grid[piece_x][piece_y]
                if c != None:
                    self.board[piece_x + x][piece_y + y] = c

        # Clear completed lines
        board2 = self.make_board()
        board2_row = 19
        for y in range(19, -1, -1):
            s = 0
            for x in range(10):
                if self.board[x][y] != None:
                    s += 1
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
                c = self.board[x][y]
                if c is None:
                    ret += " "
                else:
                    ret += str(c) # TODO: remove str() call
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

def play_min_height_game():
    b = board()
    all_pieces = list(pieces())
    while(True):
        p = random.choice(all_pieces)
        moves = list(b.valid_moves(p))
        if len(moves) == 0:
            break
        max_move = None
        max_y = -1
        for move in moves:
            y = move[2]
            if y > max_y:
                max_move = move
                max_y = y
        (r, x, y) = max_move
        b.place(r, x, y)
        print(b)
        time.sleep(.25)

if __name__ == "__main__":
    #play_random_game()
    play_min_height_game()

