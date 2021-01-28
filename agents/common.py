from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple


class SavedState:
    pass


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.full((6,7), BoardPiece(0), dtype=BoardPiece(0))


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    line = "|==============|"
    column_line = "|0 1 2 3 4 5 6 |"
    pretty_board = "\n" + line
    for j in range(5, -1, -1):
        pretty_board = pretty_board + "\n|"
        for i in range(7):
            if board[j, i] == NO_PLAYER:
                pretty_board = pretty_board + "  "
            elif board[j, i] == PLAYER1:
                pretty_board = pretty_board + "X "
            elif board[j, i] == PLAYER2:
                pretty_board = pretty_board + "O "
        pretty_board = pretty_board + "|"

    pretty_board = pretty_board + "\n" + line + "\n" + column_line
    #print(pretty_board)
    return pretty_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    rowList = pp_board.split("\n")[1:7]

    board = np.empty((0, 7), int)
    for row in rowList:
        tempRowList = [[]]
        for idx, col in enumerate(row):
            if idx < 7:
                piece = row[idx * 2 + 1]
                if piece == "O":
                    tempRowList[0].append(2)
                elif piece == "X":
                    tempRowList[0].append(1)
                else:
                    tempRowList[0].append(0)

        board = np.append(board, np.array(tempRowList), axis=0)

    return np.flip(board, axis=0)


def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy:
        backupBoard = board.copy()

    colList = board[:, action]
    for idx, piece in enumerate(colList):
        if piece == 0:
            colList[idx] = player
            break

    board[:, action] = colList
    return board


def connected_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    listOfRowsAndCols = []
    lengthOfStreak = 0
    flippedBoard = np.fliplr(board)

    for x in range(6):
        listOfRowsAndCols.append(board[x, :])
        listOfRowsAndCols.append(board[:, x])
        listOfRowsAndCols.append(np.diag(board, x))
        listOfRowsAndCols.append(np.diag(flippedBoard, x))
        listOfRowsAndCols.append(np.diag(board, (-x - 1)))
        listOfRowsAndCols.append(np.diag(flippedBoard, (-x - 1)))
    listOfRowsAndCols.append(board[:, 6])

    for rowList in listOfRowsAndCols:
        for col in rowList:
            if col == player:
                lengthOfStreak += 1
                if lengthOfStreak > 3:
                    return True
            else:
                lengthOfStreak = 0
        lengthOfStreak = 0

    return False


def column_to_be_played_for_win(board: np.ndarray, player: BoardPiece) -> PlayerAction:
    """
    Returns column the input player would win the game with when played, returns -1 if it's
    impossible with one move
    """
    # TODO: watch out for impossible moves?
    for i in range(7):
        board_cpy = board.copy()
        if connected_four(apply_player_action(board_cpy, i, player), player):
            return i                                                                # only returns the min column player could win with

    return -1


def connected_two(board: np.ndarray, player: BoardPiece) -> bool:
    listOfRowsAndCols = []
    lengthOfStreak = 0
    flippedBoard = np.fliplr(board)

    for x in range(6):
        listOfRowsAndCols.append(board[x, :])
        listOfRowsAndCols.append(board[:, x])
        listOfRowsAndCols.append(np.diag(board, x))
        listOfRowsAndCols.append(np.diag(flippedBoard, x))
        listOfRowsAndCols.append(np.diag(board, (-x - 1)))
        listOfRowsAndCols.append(np.diag(flippedBoard, (-x - 1)))
    listOfRowsAndCols.append(board[:, 6])

    for rowList in listOfRowsAndCols:
        for col in rowList:
            if col == player:
                lengthOfStreak += 1
                if lengthOfStreak > 1:
                    return True
            else:
                lengthOfStreak = 0
        lengthOfStreak = 0

    return False




def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    elif 0 not in board:
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING


def other_player(player: BoardPiece) -> BoardPiece:
    """
    for switching players. Checks which player is given and returns other
    """
    if player == PLAYER1:
        return PLAYER2
    elif player == PLAYER2:
        return PLAYER1


def move_is_possible(board: np.ndarray, action: PlayerAction) -> bool:
    """
    returns True if column is not full yet and False if it is
    """
    if board[5, action] == 0:
        return True
    else:
        return False
