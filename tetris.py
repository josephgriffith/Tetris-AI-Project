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

def epsilonGreedy(Qnet, board, piece, epsilon):
    moves = board.valid_moves(piece)
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
    all_pieces = list(pieces())
    while True:
        # Generate the next piece
        piece = b.next_piece
        (position, rotation) = ai_get_move(b, piece)
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
    while(not b.is_game_over()):
        pass
    # Train the network a number of times with that game
    pass

class rotated_piece(object):
    def __init__(self, grid, which_piece):
        self.grid = grid
        self.height = len(grid[0])
        self.width = len(grid)
        self.which_piece = which_piece

    def __str__(self):
        return str(self.grid)

    def __repr__(self):
        return "rotated_piece(" + repr(self.grid) + ")"

class piece(object):
    def __init__(self, grid, num_rotations, which_piece):
        self.grid = grid
        self.num_rotations = num_rotations
        self.which_piece = which_piece

    def rotations(self):
        # Generator that returns all the rotations for this piece.
        rotated_grid = self.grid
        for i in range(self.num_rotations):
            yield rotated_piece(rotated_grid, self.which_piece)
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
        self.all_pieces = list(pieces())
        self.next_piece = self.choose_random_piece()

    def choose_random_piece(self):
        return random.choice(self.all_pieces)

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
        # TODO: it probably shouldn't be possible to call this with a piece other than next_piece
        for piece_x in range(rotated_piece.width):
            for piece_y in range(rotated_piece.height):
                c = rotated_piece.grid[piece_x][piece_y]
                if c != 0:
                    self.board[piece_x + x][piece_y + y] = rotated_piece.which_piece

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

    def drop(self, rotated_piece, col):
        row = self.fits_row(rotated_piece, col)
        # TODO: what if row is -1?
        self.place(rotated_piece, col, row)

    def is_game_over(self):
        # TODO
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
        p = b.next_piece
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
        p = b.next_piece
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

