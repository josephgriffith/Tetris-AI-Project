import time
import random
import numpy as np
import neuralnetworks as nn
from copy import deepcopy

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
    b = Board()
    while True:
        # Generate the next piece
        (position, rotation) = ai_get_move(b)
        b.drop(rotation, position)
        b.advance_game_state()
        if b.is_game_over():
            return

def train(hiddenLayers):
    Qnet = nn.NeuralNetwork(numInputs, hiddenLayers, numOutputs)
    Qnet._standardizeT = lambda x: x # TODO: is this needed?
    Qnet._standardizeX = lambda x: x

    # TODO: do this all a number of times

    # Play a game, collecting samples
    b = Board()
    while(not b.game_over):
        pass
    # Train the network a number of times with that game
    pass

class Piece(object):
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
    inverted_red = '\033[41m'
    inverted_green = '\033[42m'
    inverted_yellow = '\033[43m'
    inverted_blue = '\033[44m'
    inverted_purple = '\033[45m'
    inverted_teal = '\033[46m'
    inverted_grey = '\033[47m'
    inverted_almostwhite = '\033[100m'

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
            'Z': Color.inverted_purple,
            ' ': Color.inverted_almostwhite
            }

    texture = '  ' if char == ' ' else '_|'
    return colors[char] + Color.bold + texture + Color.end


def pieces():
    # generator that returns all the non-rotated pieces
    # Note these are mirrored here since X is the first dimension and Y is the second.
    yield Piece(((1,),
                 (1,),
                 (1,),
                 (1,)), 2, "I")

    yield Piece(((0, 1, 0),
                 (1, 1, 1)), 4, "T")

    yield Piece(((1, 0),
                 (1, 0),
                 (1, 1)), 4, "J")

    yield Piece(((0, 1),
                 (0, 1),
                 (1, 1)), 4, "L")

    yield Piece(((1, 1),
                 (1, 1)), 1, "O")

    yield Piece(((1, 0),
                 (1, 1),
                 (0, 1)), 2, "Z")

    yield Piece(((0, 1),
                 (1, 1),
                 (1, 0)), 2, "S")

class Board(object):
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = self.make_board()
        self.upcoming = [[None]*4 for i in range(4)]
        self.all_pieces = list(pieces())
        self.cleared = [0]*height
        self.game_over = False
        self.advance_game_state()
        self.paintPiece(self.upcoming)

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

    def paintPiece(self, canvas, rotation=0, x=0, y=0):
        (width, height) = self.next_piece.get_dimensions(rotation)
        grid = self.next_piece.get_rotated_grid(rotation)
        for piece_x in range(width):
            for piece_y in range(height):
                c = grid[piece_x][piece_y]
                if c != 0:
                    canvas[piece_x + x][piece_y + y] = self.next_piece.which_piece

    def place(self, rotation, x, y):
        self.paintPiece(self.board, rotation, x, y)
        # We can set the 'cleared' variable even before the game state is advanced.
        self.cleared = self.find_completed_lines()

    def advance_game_state(self):
        self.clear_completed_lines_in_board()

        # Choose next piece
        self.next_piece = self.choose_random_piece()

        # Find all valid moves
        self.valid_moves = []
        for rot in range(self.next_piece.num_rotations):
            for col in range(self.width):
                row = self.fits_row(rot, col)
                if row >= 0:
                    self.valid_moves.append((rot, col, row))

        # Check for game over
        if len(self.valid_moves) == 0:
            self.game_over = True
            return

        # Clear upcoming (after game over so we don't render the upcoming piece if the game is over)
        if not self.game_over:
            self.upcoming = [[None]*4 for i in range(4)]
            self.paintPiece(self.upcoming)

    def find_completed_lines(self):
        # Detect completed lines
        cleared = [0]*self.height
        for y in range(self.height - 1, -1, -1):
            s = 0
            for x in range(self.width):
                if self.board[x][y] != None:
                    s += 1
            if s == self.width: # If we cleared a line
                cleared[y] = 1
        return cleared

    def clear_completed_lines_in_board(self):
        new_row = self.height - 1
        for y in range(self.height - 1, -1, -1):
            if new_row != y:
                for x in range(self.width):
                    self.board[x][new_row] = self.board[x][y]
            if self.cleared[y] == 0: # If we didn't clear a line
                new_row -= 1
        self.cleared = [0]*self.height

    def drop(self, rotation, col):
        row = self.fits_row(rotation, col)
        if row == -1: # Invalid move requested
            self.game_over = True
        else:
            self.place(rotation, col, row)

    def thing(self, clear=False):
       ret = "+" + "--" * (self.width) + "+\n"
       for y in range(self.height):
           ret += "|"
           for x in range(self.width):
               if clear and self.cleared[y] == 1:
                   ret += colorize(' ')
               else:
                   ret += colorize(self.board[x][y])
           if y > 3 or clear:
               ret += "|\n"
           else:
               ret += "|\t\t" + \
                       colorize(self.upcoming[0][y]) + \
                       colorize(self.upcoming[1][y]) + \
                       colorize(self.upcoming[2][y]) + \
                       colorize(self.upcoming[3][y]) + "\n"
       ret += "+" + "--" * (self.width) + "+" + "\n"
       return ret

    def __str__(self):
       ret = ''
       ret += self.thing()
       if self.cleared.count(1) > 0:
           #TODO: check that multiple clear lines works! -- probably need a way to manually pick moves first
           ret += '\n' + self.thing(True)
       return ret

def play_random_game():
    b = Board()
    while(True):
        if b.game_over:
            break
        (r, x, y) = random.choice(b.valid_moves)
        b.place(r, x, y)
        if b.cleared.count(1) > 0:
            print(b)
            time.sleep(.25)
        b.advance_game_state()
        print(b)
        time.sleep(.25)

def play_min_height_game():
    b = Board()
    print(b)
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
        if b.cleared.count(1) > 0:
            print(b)
            time.sleep(.5)
        b.advance_game_state()
        print(b)
        time.sleep(.5)

def displayAllRotations():
    all_pieces = list(pieces())
    for i in all_pieces:
        b = Board()
        x, y = 0, 0
        for rot in range(i.num_rotations):
            b.next_piece = i
            b.place(rot, x, y)
            b.advance_game_state()
            y += 5
        print(b)

if __name__ == "__main__":
    #displayAllRotations()
    #play_random_game()
    play_min_height_game()
