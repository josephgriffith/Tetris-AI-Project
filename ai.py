import time
import random
import numpy as np
import neuralnetworks as nn
import tetris
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
        move = board.valid_moves[np.argmin(qs)]
        Q = np.min(qs)
    return move, Q

def train(nReps, hiddenLayers, epsilon, epsilonDecayFactor, nTrainIterations, nReplays):
    # The inputs to the neural network are:
    #   width * height 0/1 values for the board
    #   7 inputs for which piece we're placing
    #   A column to place the piece in - 10 values
    #   A piece rotation - 4 values
    # The output from the neural network is:
    #   A single number to represent the estimated number of moves to game over.
    boardWidth = 10
    boardHeight = 20
    numDataCols = boardWidth * boardHeight + 7 + 10 + 4
    Qnet = nn.NeuralNetwork(numDataCols, hiddenLayers, 1)
    Qnet._standardizeT = lambda x: x
    Qnet._unstandardizeT = lambda x: x

    outcomes = np.zeros(nReps)
    for rep in range(nReps):
        if rep > 0:
            epsilon *= epsilonDecayFactor

        # Play a game, collecting samples
        samples = []
        samplesNextStateForReplay = []
        board = tetris.Board(boardWidth, boardHeight)
        move, _ = epsilonGreedy(Qnet, board, epsilon)
        done = False
        step = 0
        while not done:
            step += 1

            if step > 100:
                return Qnet, outcomes

            # print(board)

            newBoard = deepcopy(board)
            newBoard.make_move(move)

            if newBoard.game_over:
                done = True
                Qnext = 0
                outcomes[rep] = step
                print("Played game", rep, ", lasted for", step, "moves, epsilon is", epsilon)
            else:
                moveNext, Qnext = epsilonGreedy(Qnet, newBoard, epsilon)

            r = -1
            stateRepresentation = board.getStateRepresentation()
            moveRepresentation = board.getMoveRepresentation(move)
            # fullRep = (*stateRepresentation, *moveRepresentation)
            # It's possible to see the same board state twice.
            # If that happens, we should only keep the one furthest from game over,
            # since that represents the true game length from that state.
            # if fullRep in samples:
            #     (_, existingQnext) = samples[fullRep]
            #     Qnext = min(Qnext, existingQnext)
            # samples[fullRep] = (r, Qnext)
            samples.append([*stateRepresentation, *moveRepresentation, r, Qnext])
            samplesNextStateForReplay.append([*newBoard.getStateRepresentation(), *newBoard.getMoveRepresentation(moveNext)])

            move = moveNext
            board = newBoard

        # Convert samples to an array.
        # samples_ary = []
        # for key, value in samples.items():
        #     samples_ary.append([*key, *value])
        # samples = np.array(samples_ary)
        samples = np.array(samples)
        #print(samples[:, numDataCols+1])
        #print(samples)
        X = samples[:, :numDataCols]
        T = samples[:, numDataCols:numDataCols+1] + samples[:,numDataCols+1:numDataCols+2]

        # We know how many moves were remaining at each state of the game, since we can count from the end
        # of the game.  So let's use that data to train.
        # T = np.array(range(len(samples)-1, -1, -1))


        Qnet.train(X, T, nTrainIterations, verbose=False)
        # print(Qnet.W[:,0])

        #print(Qnet.getErrorTrace())

        # Experience replay
        samplesNextStateForReplay = np.array(samplesNextStateForReplay)
        for replay in range(nReplays):
            QnextNotZero = samples[:, numDataCols+1] != 0
            samples[QnextNotZero, numDataCols+1:numDataCols+2] = Qnet.use(samplesNextStateForReplay[QnextNotZero,:])
            T = samples[:, numDataCols:numDataCols+1] + samples[:,numDataCols+1:numDataCols+2]
            Qnet.train(X, T, nTrainIterations, verbose=False)

    return Qnet, outcomes

def choose_best_move(board, moveScoreF):
    """ Choose the best move based on a function that returns a score for each move.  The highest-scoring move will be chosen.
    MoveScoreF takes a move and returns a score. """
    best_score_move = None
    best_score = None
    for move in board.valid_moves:
        score = moveScoreF(board, move)
        if best_score is None or score > best_score:
            best_score = score
            best_score_move = move
    return best_score_move

def randomMoveStrategy(board):
    return random.choice(board.valid_moves)

def minHeightStrategy(board):
    return choose_best_move(board, lambda board, move: move[2])

def holesAndHeightStrategy(board):
    def holesAndHeightScore(board, move):
        b2 = deepcopy(board) # 63% of the time
        b2.make_move(move)
        holes = b2.count_holes()
        y = move[2]
        height_factor = .5
        holes_factor = .5
        return height_factor * y - holes_factor * holes
    return choose_best_move(board, holesAndHeightScore)

def play_ai_parameterized(nReps, hiddenLayers, epsilon, epsilonDecayFactor, nTrainIterations, nReplays):
    print("nReps", nReps, "hiddenLayers", hiddenLayers, "epsilon", epsilon, "epsilonDecayFactor", epsilonDecayFactor, "nTrainIterations", nTrainIterations, "nReplays", nReplays, ": ", end="", flush=True)
    startTime = time.time()
    (Qnet, outcomes) = train(nReps=nReps,
            hiddenLayers=hiddenLayers,
            epsilon=epsilon,
            epsilonDecayFactor=epsilonDecayFactor,
            nTrainIterations=nTrainIterations,
            nReplays=nReplays)
    endTime = time.time()
    elapsedTime = endTime - startTime
    longest_game = max(outcomes)
    average_game = sum(outcomes) / len(outcomes)
    print("longest", longest_game, "average", average_game, "elapsedTime", elapsedTime)

def play_several_ai_games():
    for i in range(1, 10):
        hiddenLayers = [1] * i
        nReps = 1000
        epsilon = 1
        epsilonDecayFactor = .96
        nTrainIterations = 1
        nReplays = 0

        play_ai_parameterized(nReps, hiddenLayers, epsilon, epsilonDecayFactor, nTrainIterations, nReplays)

def play_ai_game(reps=100, layers=[50, 20, 10, 2, 10, 20, 50], eps=1, decay=.95, iterations=1, replays=1):
    (Qnet, outcomes) = train(nReps=reps,
            #hiddenLayers=[20, 10, 10, 20],
            hiddenLayers=layers,
            epsilon=eps,
            epsilonDecayFactor=decay,
            nTrainIterations=iterations,
            nReplays=replays)
    print(",".join(str(int(x)) for x in outcomes))

    def AIStrategy(board):
        move, Q = epsilonGreedy(Qnet, board, 0)
        return move

    tetris.Board().play_game(AIStrategy, True, .25)

def play_several_min_holes_games():
    num_games_per_test = 5
    for i in np.linspace(.3, .7, num=10):
        outcomes = []
        for n in range(num_games_per_test):
            outcome = tetris.Board().play_game(holesAndHeightStrategy)
            outcomes.append(outcome)
            print("game", n, "outcome", outcome)
        print("i", i, "min", min(outcomes), "max", max(outcomes), "avg", sum(outcomes)/len(outcomes), flush=True)

if __name__ == "__main__":
    #tetris.Board().play_game(randomMoveStrategy, True, .25)
    #tetris.Board().play_game(minHeightStrategy, True, .25)
    tetris.Board().play_game(holesAndHeightStrategy, True, .1)

    #play_ai_game()

    #np.set_printoptions(suppress=True) # Turn off scientific notation
    #np.set_printoptions(threshold=np.inf) # Print the whole outcomes array

    #play_several_ai_games()
    #play_several_min_holes_games()
