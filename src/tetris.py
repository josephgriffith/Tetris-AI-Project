import random
import numpy as np

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

    @staticmethod
    def choose_random_piece():
        return random.choice(Piece.pieces())

    @staticmethod
    def pieces():
        # generator that returns all the non-rotated pieces
        # Note these are mirrored here since X is the first dimension and Y is the second.
        I = Piece(((1,),
                     (1,),
                     (1,),
                     (1,)), 2, "I")

        T = Piece(((0, 1, 0),
                     (1, 1, 1)), 4, "T")

        J = Piece(((1, 0),
                     (1, 0),
                     (1, 1)), 4, "J")

        L = Piece(((0, 1),
                     (0, 1),
                     (1, 1)), 4, "L")

        O = Piece(((1, 1),
                     (1, 1)), 1, "O")

        Z = Piece(((1, 0),
                     (1, 1),
                     (0, 1)), 2, "Z")

        S = Piece(((0, 1),
                     (1, 1),
                     (1, 0)), 2, "S")

        return [I, T, J, L, O, Z, S]
        #return [I, O]

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

    @staticmethod
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

class Board(object):
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = [[None] * self.height for i in range(self.width)]
        self.upcoming = [[None]*4 for i in range(4)]
        self.cleared = [0]*height
        self.game_over = False

        self.advance_game_state()
        self.paintPiece(self.upcoming)

    def getStateRepresentation(self):
        # 10 column heights, 7 booleans for which piece is next
        data = [0] * self.height * self.width
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y] is not None:
                    data[x * y] = 1

        # cols = [self.height] * 10
        # for i in range(self.width):
        #     for j in range(self.height):
        #         if self.board[i][j] != None:
        #             cols[i] = j
        #             break

        # # Normalize height to be 0-1
        # for i in range(self.width):
        #     cols[i] /= self.height
        #     cols[i] = 1 - cols[i]

        piece = [0] * 7
        d = {"I": 0, "O": 1, "L": 2, "J": 3, "S": 4, "Z": 5, "T": 6}
        piece[d[self.next_piece.which_piece]] = 1

        return [*data, *piece]

    def getMoveRepresentation(self, move):
        (rot, col, row) = move
        # Output is 10 booleans for col, 4 outputs for rot
        colOut = [0] * self.width
        colOut[col] = 1

        rotOut = [0] * 4
        rotOut[rot] = 1

        return [*colOut, *rotOut]

    def fits_row(self, rotation, col):
        # Returns the one and only one row that the piece must be placed at in this column.
        (width, height) = self.next_piece.get_dimensions(rotation)
        if width + col > self.width:
            return -1
        grid = self.next_piece.get_rotated_grid(rotation)
        for offset_y in range(self.height + 1):
            for piece_y in range(height-1, -1, -1):
                if piece_y + offset_y == self.height:
                    # Off the bottom of the board
                    return offset_y - 1;
                for piece_x in range(width):
                    if self.board[piece_x + col][piece_y + offset_y] is not None and grid[piece_x][piece_y] != 0:
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
        self.next_piece = Piece.choose_random_piece()

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

    def board_to_string(self, clear=False, print_upcoming=True):
        ret = "+" + "--" * (self.width) + "+\n"
        for y in range(self.height):
            ret += "|"
            for x in range(self.width):
                if clear and self.cleared[y] == 1:
                    ret += Color.colorize(' ')
                else:
                    ret += Color.colorize(self.board[x][y])
            if y > 3 or print_upcoming == False:
                ret += "|\n"
            else:
                ret += "|\t\t"
                for i in range(4):
                    ret += Color.colorize(self.upcoming[i][y])
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

    def count_holes(self):
        num_holes = 0
        for x in range(self.width):
            counting = False
            for y in range(0, self.height):
                if self.board[x][y] is not None:
                    counting = True
                else:
                    if counting:
                        num_holes += 1
        return num_holes

def displayAllRotations():
    all_pieces = list(Piece.pieces())
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
    displayAllRotations()
