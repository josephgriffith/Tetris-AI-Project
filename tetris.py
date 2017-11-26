import time
import random
import numpy as np
import neuralnetworks as nn
from copy import deepcopy

# Goal: the AI should play a long game.
# States which are closer to losing should have lower scores.
# So, let's define a lost game to have value 0.  Then the state before a lost game has value 1, etc.
# Then our use() function should choose the highest-value move.
# The more we train, the higher the value for the initial state should be.

# Much of this code is taken and/or adapted from http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/21%20Reinforcement%20Learning%20with%20a%20Neural%20Network%20as%20the%20Q%20Function.ipynb

def epsilonGreedy(Qnet, board, epsilon):
    if np.random.uniform() < epsilon: # random move
        move = random.choice(board.valid_moves)
        if Qnet.Xmeans is None:
            # Qnet is not initialized yet
            Q = 0
        else:
            stateMoveRepresentation = board.getStateRepresentation() + board.getMoveRepresentation(move)
            Q = Qnet.use(stateMoveRepresentation)
    else: # greedy move
        qs = []
        for m in board.valid_moves:
            stateMoveRepresentation = board.getStateRepresentation() + board.getMoveRepresentation(m)
            qs.append(Qnet.use(stateMoveRepresentation) if Qnet.Xmeans is not None else 0)
        move = board.valid_moves[np.argmax(qs)]
        Q = np.max(qs)
    return move, Q

def train(nReps, hiddenLayers, epsilon, epsilonDecayFactor, nTrainIterations, nReplays):
    # The inputs to the neural network are:
    #   10 column heights
    #   7 inputs for which piece we're placing
    #   A column to place the piece in - 10 values
    #   A piece rotation - 4 values
    # The output from the neural network is:
    #   A single number to represent the estimated number of moves to game over.
    Qnet = nn.NeuralNetwork(31, hiddenLayers, 1)
    Qnet._standardizeT = lambda x: x
    Qnet._standardizeX = lambda x: x

    outcomes = np.zeros(nReps)
    for rep in range(nReps):
        if rep > 0:
            epsilon *= epsilonDecayFactor

        # Play a game, collecting samples
        samples = []
        # samplesNextStateForReplay = [] # TODO: this information is duplicated with samples...
        board = Board()
        move, _ = epsilonGreedy(Qnet, board, epsilon)
        done = False
        step = 0
        while not done:
            step += 1

            # TODO: move contains row and probably shouldn't

            # print(board)

            newBoard = deepcopy(board) # TODO: make a function that returns the new state without modifying the current state, instead of deepcopy
            newBoard.make_move(move)

            if newBoard.game_over:
                done = True
                Qnext = 0
                outcomes[rep] = step
                print("Played game", rep, ", lasted for", step, "moves, epsilon is", epsilon)
            else:
                moveNext, Qnext = epsilonGreedy(Qnet, newBoard, epsilon)

            r = 1 # We're trying to maximize the number of moves before game over, so we want r to be positive.
            stateRepresentation = board.getStateRepresentation()
            moveRepresentation = board.getMoveRepresentation(move)
            samples.append([*stateRepresentation, *moveRepresentation, r, Qnext])
            # samplesNextStateForReplay.append([*newBoard.getStateRepresentation(), *newBoard.getMoveRepresentation(moveNext)])

            move = moveNext
            board = newBoard

        samples = np.array(samples)
        print(samples[:, 32])
        X = samples[:, :31]
        # T = samples[:, 31:32] + samples[:,32:33]

        # We know how many moves were remaining at each state of the game, since we can count from the end
        # of the game.  So let's use that data to train.
        T = np.array(range(len(samples)-1, -1, -1))
        Qnet.train(X, T, nTrainIterations, verbose=False)

        # Experience replay
        # samplesNextStateForReplay = np.array(samplesNextStateForReplay)
        for replay in range(nReplays):
            # QnextNotZero = samples[:, 32] != 0
            # samples[QnextNotZero, 31:32] = Qnet.use(samplesNextStateForReplay[QnextNotZero,:])
            #T = samples[:, 31:32] + samples[:,32:33]
            Qnet.train(X, T, nTrainIterations, verbose=False)

    return Qnet, outcomes

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

    def set_next_piece(self, piece):
        self.next_piece = piece
        self.upcoming = [[None]*4 for i in range(4)]
        self.paintPiece(self.upcoming)

    def getStateRepresentation(self):
        # 10 column heights, 7 booleans for which piece is next
        cols = [self.height] * 10
        for i in range(self.width):
            for j in range(self.height):
                if self.board[i][j] != None:
                    cols[i] = j
                    break

        # Normalize height to be 0-1
        for i in range(self.width):
            cols[i] /= self.height

        piece = [0] * 7
        d = {"I": 0, "O": 1, "L": 2, "J": 3, "S": 4, "Z": 5, "T": 6}
        piece[d[self.next_piece.which_piece]] = 1

        return [*cols, *piece]

    def getMoveRepresentation(self, move):
        (rot, col, row) = move
        # Output is 10 booleans for col, 4 outputs for rot
        colOut = [0] * self.width
        colOut[col] = 1

        rotOut = [0] * 4
        rotOut[rot] = 1

        return [*colOut, *rotOut]

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

    def make_move(self, move):
        self.place(*move)
        self.advance_game_state()

    def place(self, rotation, x, y):
        self.paintPiece(self.board, rotation, x, y)

        # Detect completed lines; put the result in self.cleared
        # We can do this even before the game state is advanced.  This allows __str__ to highlight the cleared rows.
        self.cleared = [0]*self.height
        for y in range(self.height - 1, -1, -1):
            s = 0
            for x in range(self.width):
                if self.board[x][y] != None:
                    s += 1
            if s == self.width: # If we cleared a line
                self.cleared[y] = 1

    def advance_game_state(self):
        # Choose next piece
        self.next_piece = self.choose_random_piece()

        # Clear cleared lines in the board
        new_row = self.height - 1
        for y in range(self.height - 1, -1, -1):
            if new_row != y:
                for x in range(self.width):
                    self.board[x][new_row] = self.board[x][y]
            if self.cleared[y] == 0: # If we didn't clear a line
                new_row -= 1
        for y in range(0, new_row+1):
            for x in range(self.width):
                self.board[x][y] = None
        self.cleared = [0]*self.height

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

        # Clear upcoming (after game over so we don't render the upcoming piece if the game is over)
        self.upcoming = [[None]*4 for i in range(4)]

        # Draw upcoming
        if not self.game_over:
            self.paintPiece(self.upcoming)

    def drop(self, rotation, col):
        row = self.fits_row(rotation, col)
        if row == -1: # Invalid move requested
            self.game_over = True
        else:
            self.place(rotation, col, row)

    def board_to_string(self, clear=False, print_upcoming=True):
        ret = "+" + "--" * (self.width) + "+\n"
        for y in range(self.height):
            ret += "|"
            for x in range(self.width):
                if clear and self.cleared[y] == 1:
                    ret += colorize(' ')
                else:
                    ret += colorize(self.board[x][y])
            if y > 3 or print_upcoming == False:
                ret += "|\n"
            else:
                ret += "|\t\t"
                for i in range(4):
                    ret += colorize(self.upcoming[i][y])
                ret += "\n"
        ret += "+" + "--" * (self.width) + "+" + "\n"
        return ret

    def __str__(self):
        ret = ''
        if self.cleared.count(1) > 0:
            ret += self.board_to_string(print_upcoming=False)
            ret += '\n'
            ret += self.board_to_string(True, False)
        else:
            ret += self.board_to_string(print_upcoming=True)
        return ret

def play_random_game():
    b = Board()
    numMoves = 0
    while not b.game_over:
        numMoves += 1
        (r, x, y) = random.choice(b.valid_moves)
        b.place(r, x, y)
        if b.cleared.count(1) > 0:
            print(b)
            time.sleep(.25)
        b.advance_game_state()
        print(b)
        time.sleep(.25)
    print(numMoves, "moves")

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
            time.sleep(.25)
        b.advance_game_state()
        print(b)
        time.sleep(.25)

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

def play_ai_game():
    (Qnet, outcomes) = train(nReps=1000,
            hiddenLayers=[20, 10, 2, 10, 20],
            epsilon=1,
            epsilonDecayFactor=.995,
            nTrainIterations=5,
            nReplays=5)
    print(outcomes)

    # Play a game using the trained AI
    b = Board()
    numMoves = 0
    while not b.game_over:
        move, Q = epsilonGreedy(Qnet, b, 0)
        b.make_move(move)
        numMoves += 1
        print(b)
        time.sleep(.25)
    print(numMoves, "moves")

if __name__ == "__main__":
    #displayAllRotations()
    #play_random_game()
    #play_min_height_game()
    play_ai_game()
