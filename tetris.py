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
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    purple = '\033[95m'
    teal = '\033[96m'
    grey = '\033[97m'
    black = '\033[98m'
    #black = '\033[99m'

    end = '\033[0m'
    bold = '\033[1m'

    #char = chr(35)          # octothorpe
    char = chr(164)         # spiky circle
    #char = chr(449)         # ||                   ERRORS in console
    I = red + bold + char + end
    T = green + bold + char + end
    L = yellow + bold + char + end
    J = blue + bold + char + end
    O = grey + bold + char + end
    S = teal + bold + char + end
    Z = purple + bold + char + end

    # Note these are mirrored here since X is the first dimension and Y is the second.
    yield piece(((I,),
                 (I,),
                 (I,),
                 (I,)), 2)

    yield piece(((None, T, None),
                 (T, T, T)), 4)

    yield piece(((J, None),
                 (J, None),
                 (J, J)), 4)

    yield piece(((None, L),
                 (None, L),
                 (L, L)), 4)

    yield piece(((O, O),
                 (O, O)), 1)

    yield piece(((Z, None),
                 (Z, Z),
                 (None, Z)), 2)

    yield piece(((None, S),
                 (S, S),
                 (S, None)), 2)

class board(object):
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = self.make_board()

    def make_board(self):
        return [[None] * self.height for i in range(self.width)]

    def fits_row(self, rotated_piece, col):
        # Returns the one and only one row that the piece must be placed at in this column.
        if rotated_piece.width + col > self.width:
            return -1
        for offset_y in range(self.height + 1):
            for piece_x in range(rotated_piece.width):
                for piece_y in range(rotated_piece.height):
                    if piece_y + offset_y == self.height:
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
        board2_row = self.height - 1
        for y in range(self.height - 1, -1, -1):
            s = 0
            for x in range(self.width):
                if self.board[x][y] != None:
                    s += 1
            if s != self.width:
                for x in range(self.width):
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
            for col in range(self.width):
                row = self.fits_row(rot, col)
                if row >= 0:
                    yield (rot, col, row)

    def __str__(self):
        ret = "+" + "-" * self.width + "+" + "\n"
        for y in range(self.height):
            ret += "|"
            for x in range(self.width):
                c = self.board[x][y]
                if c is None:
                    ret += " "
                else:
                    ret += c
            ret += "|\n"
        ret += "+" + "-" * self.width + "+" + "\n"
        return ret

    #spaced print
    #def __str__(self):
    #   ret = "+" + "--" * (self.width-1) + "-+" + "\n"
    #   for y in range(self.height):
    #       ret += "|"
    #       for x in range(self.width):
    #           c = self.board[x][y]
    #           if c is None:
    #               ret += "  "
    #           else:
    #               ret += c + " "
    #       ret += "\b|\n"
    #   ret += "+" + "--" * (self.width-1) + "-+" + "\n"
    #   return ret

    #grid print
    #def __str__(self):
    #    under = '\033[4m'
    #    end = '\033[0m'
    #    ret = under + "  " * self.width + " " + "\n"
    #    for y in range(self.height):
    #        ret += under + "|"
    #        for x in range(self.width):
    #            c = self.board[x][y]
    #            if c is None:
    #                ret += under + " |"
    #            else:
    #                ret += under + c + under + "|"
    #        ret += "\n"
    #    return ret

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

def displayAllRotations():
    all_pieces = list(pieces())
    for i in all_pieces:
        b = board()
        x, y = 0, 0
        for rot in i.rotations():
            b.place(rot, x, y)
            y += 5
        print(b)

if __name__ == "__main__":
    #displayAllRotations()
    #play_random_game()
    play_min_height_game()

