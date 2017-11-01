
class rotated_piece(object):
    def is_outside_board(x, y):
        pass

class piece(object):
    def __init__(self, grid, num_rotations):
        pass

    def rotations(self):
        # Generator that returns all the rotations for this piece.
        pass

def pieces():
    # generator that returns all the non-rotated pieces
    yield piece([[1],
                 [1],
                 [1],
                 [1]], 2)

    yield piece([[0, 1, 0],
                 [1, 1, 1]], 4)

    yield piece([[1, 0],
                 [1, 0],
                 [1, 1]], 4)

    yield piece([[0, 1],
                 [0, 1],
                 [1, 1]], 4)

    yield piece([[1, 1],
                 [1, 1]], 1)

    yield piece([[1, 0],
                 [1, 1],
                 [0, 1]], 2)

    yield piece([[0, 1],
                 [1, 1],
                 [1, 0]], 2)

class board(object):
    def fits_row(self, rotated_piece, col):
        # Returns the one and only one row that the piece must be placed at in this column.
        pass

    def fits(self, rotated_piece, x, y):
        pass

    def is_game_over(self):
        pass

    def valid_moves(self, piece):
        # generator that returns all the valid moves for the given piece
        pass
