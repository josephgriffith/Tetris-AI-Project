import time
import random
import numpy as np
import neuralnetworks as nn

# Goal: the AI should play a long game.
# We'll have a train function to train the neural net.
# The train function will play games of tetris and observe the outcome.
# States which are closer to losing should have lower scores.
# So, let's define a lost game to have value 0.  Then the state before a lost game has value 1, etc.
# Then our use() function should choose the highest-value move.
# The more we train, the higher the value for the initial state should be.
# The inputs to the neural network are:
#   10 column heights
#   7 inputs for which piece we're placing
# The outputs from the neural network are:
#   A column - 10 outputs
#   A piece rotation - 4 outputs

# What do we do if the AI tries to make an illegal move?

# Much of this code is taken and/or adapted from http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/21%20Reinforcement%20Learning%20with%20a%20Neural%20Network%20as%20the%20Q%20Function.ipynb

def epsilonGreedy(Qnet, board, epsilon):
    moves = board.valid_moves
    if np.random.uniform() < epsilon: # random move
        move = random.choice(moves)
        if Qnet.Xmeans is None:
            # Qnet is not initialized yet
            Q = 0
        else:
            Q = Qnet.use(something)
    else: # greedy move
        qs = []
        for m in moves:
            qs.append(Qnet.use(something) if Qnet.Xmeans is not None else 0)
        move = moves[np.argmax(qs)]
        Q = np.max(qs)
    return move, Q

def play_ai_game():
    b = board()
    while True:
        # Generate the next piece
        (position, rotation) = ai_get_move(b)
        b.drop(rotation, position)
        if b.is_game_over():
            return

def train(hiddenLayers):
    Qnet = nn.NeuralNetwork(numInputs, hiddenLayers, numOutputs)
    Qnet._standardizeT = lambda x: x # TODO: is this needed?
    Qnet._standardizeX = lambda x: x

    # TODO: do this all a number of times

    # Play a game, collecting samples
    b = board()
    while(not b.game_over):
        pass
    # Train the network a number of times with that game
    pass

class piece(object):
    def __init__(self, grid, num_rotations, which_piece):
        self.grid = grid
        self.num_rotations = num_rotations
        self.which_piece = which_piece
        self.height = len(grid[0])
        self.width = len(grid)

    def get_dimensions(self, rotation):
        if rotation % 2 == 0:
            return (self.width, self.height)
        else:
            return (self.height, self.width)

    def get_rotated_grid(self, rotation):
        return np.rot90(self.grid, rotation)

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
        self.all_pieces = list(pieces())
        self.next_piece = self.choose_random_piece()
        self.valid_moves = list(self.find_valid_moves())
        self.game_over = False

    def choose_random_piece(self):
        return random.choice(self.all_pieces)

    def make_board(self):
        return [[None] * self.height for i in range(self.width)]

    def fits_row(self, rotation, col):
        # Returns the one and only one row that the piece must be placed at in this column.
        (width, height) = self.next_piece.get_dimensions(rotation)
        if width + col > self.width:
            return -1
        grid = self.next_piece.get_rotated_grid(rotation)
        for offset_y in range(self.height + 1):
            for piece_x in range(width):
                for piece_y in range(height):
                    if piece_y + offset_y == self.height:
                        # Off the bottom of the board
                        return offset_y - 1;
                    if self.board[piece_x + col][piece_y + offset_y] != None and grid[piece_x][piece_y] != 0:
                        # Did not fit
                        return offset_y-1

    def place(self, rotation, x, y):
        (width, height) = self.next_piece.get_dimensions(rotation)
        grid = self.next_piece.get_rotated_grid(rotation)
        for piece_x in range(width):
            for piece_y in range(height):
                c = grid[piece_x][piece_y]
                if c != 0:
                    self.board[piece_x + x][piece_y + y] = self.next_piece.which_piece

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

        # Choose next piece
        self.next_piece = self.choose_random_piece()
        self.valid_moves = list(self.find_valid_moves())
        if len(self.valid_moves) == 0:
            self.game_over = True

    def drop(self, rotation, col):
        row = self.fits_row(rotation, col)
        if row == -1: # Invalid move requested
            self.game_over = True
        else:
            self.place(rotation, col, row)

    def find_valid_moves(self):
        # generator that returns all the valid moves for the current state
        for rot in range(self.next_piece.num_rotations):
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
    while(True):
        if b.game_over:
            break
        (r, x, y) = random.choice(b.valid_moves)
        b.place(r, x, y)
        print(b)
        time.sleep(.25)

def play_min_height_game():
    b = board()
    while(True):
        if b.game_over:
            break
        max_move = None
        max_y = -1
        for move in b.valid_moves:
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
        for rot in range(i.num_rotations):
            b.next_piece = i
            b.place(rot, x, y)
            y += 5
        print(b)

if __name__ == "__main__":
    #displayAllRotations()
    #play_random_game()
    play_min_height_game()

