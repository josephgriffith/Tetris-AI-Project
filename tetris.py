import time
import random

# Goal: the AI should play a long game.
# We'll have a train function to train the neural net.
# The train function will play games of tetris and observe the outcome.
# States which are closer to losing should have lower scores.
# So, let's define a lost game to have value 0.  Then the state before a lost game has value 1, etc.
# Then our use() function should choose the highest-value move.
# The more we train, the higher the value for the initial state should be.
# The inputs to the neural network are:
#   10 column heights
#   6 zeroes and 1 one to represent which piece we're placing
# The outputs from the neural network are:
#   A column, 0-9 -- is this one output or 10?
#   A piece rotation

# What do we do if the AI tries to make an illegal move?

def play_ai_game():
    b = board()
    while True:
        # Generate the next piece
        piece = choose_random_piece()
        (position, rotation) = ai_get_move(b, piece)
        b.drop(rotation, position)
        if b.is_game_over():
            return

def train():
    # Play a game, collecting samples
    # Train the network a number of times with that game
    pass

class rotated_piece(object):
    def __init__(self, grid, display_str):
        self.grid = grid
        self.height = len(grid[0])
        self.width = len(grid)
        self.display_str = display_str

    def __str__(self):
        return str(self.grid)
    def __repr__(self):
        return "rotated_piece(" + repr(self.grid) + ")"

class piece(object):
    def __init__(self, grid, num_rotations, display_str):
        self.grid = grid
        self.num_rotations = num_rotations
        self.display_str = display_str

    def rotations(self):
        # Generator that returns all the rotations for this piece.
        rotated_grid = self.grid
        for i in range(self.num_rotations):
            yield rotated_piece(rotated_grid, self.display_str)
            rotated_grid = list(zip(*rotated_grid[::-1]))

class Color(object):
    """ Enum to keep track of console color codes. """
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    purple = '\033[95m'
    teal = '\033[96m'
    grey = '\033[97m'
    black = '\033[98m'
    #black = '\033[99m'
    inverted_red = '\033[41m'
    inverted_green = '\033[42m'
    inverted_yellow = '\033[43m'
    inverted_blue = '\033[44m'
    inverted_purple = '\033[45m'
    inverted_teal = '\033[46m'
    inverted_grey = '\033[47m'

    end = '\033[0m'
    bold = '\033[1m'

def colorize(char):
    if char == None:
        return '  '

    colors = {
            'I': Color.inverted_red,
            'T': Color.inverted_green,
            'L': Color.inverted_yellow,
            'J': Color.inverted_blue,
            'O': Color.inverted_grey,
            'S': Color.inverted_teal,
            'Z': Color.inverted_purple
            }

    return colors[char] + Color.bold + '_|' + Color.end


def pieces():
    # generator that returns all the non-rotated pieces
    # Note these are mirrored here since X is the first dimension and Y is the second.
    yield piece(((1,),
                 (1,),
                 (1,),
                 (1,)), 2, "I")

    yield piece(((0, 1, 0),
                 (1, 1, 1)), 4, "T")

    yield piece(((1, 0),
                 (1, 0),
                 (1, 1)), 4, "J")

    yield piece(((0, 1),
                 (0, 1),
                 (1, 1)), 4, "L")

    yield piece(((1, 1),
                 (1, 1)), 1, "O")

    yield piece(((1, 0),
                 (1, 1),
                 (0, 1)), 2, "Z")

    yield piece(((0, 1),
                 (1, 1),
                 (1, 0)), 2, "S")

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
                    if self.board[piece_x + col][piece_y + offset_y] != None and rotated_piece.grid[piece_x][piece_y] != 0:
                        # Did not fit
                        return offset_y-1

    def place(self, rotated_piece, x, y):
        for piece_x in range(rotated_piece.width):
            for piece_y in range(rotated_piece.height):
                c = rotated_piece.grid[piece_x][piece_y]
                if c != 0:
                    self.board[piece_x + x][piece_y + y] = rotated_piece.display_str

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
       ret = "+" + "--" * (self.width) + "+" + "\n"
       for y in range(self.height):
           ret += "|"
           for x in range(self.width):
               ret += colorize(self.board[x][y])
           ret += "|\n"
       ret += "+" + "--" * (self.width) + "+" + "\n"
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

