
class rotated_piece(object):
    def __init__(self, grid):
        self.grid = grid

    def is_outside_board(x, y):
        pass

    def __str__(self):
        return str(self.grid)

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

if __name__ == "__main__":
    for p in pieces():
        for r in p.rotations():
            print(r)
