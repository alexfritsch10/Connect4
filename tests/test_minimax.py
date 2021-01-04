import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, pretty_print_board
from agents.agent_minimax.gameState import GameState

class Tests:

    board0 = np.zeros((6, 7), BoardPiece)       #empty board
    board1 = np.ones((6, 7), BoardPiece)        #board with X everywhere

    board2 = np.zeros((6, 7), BoardPiece)       #possible played board
    board2[0, 1] = PLAYER2
    board2[0, 2] = PLAYER2
    board2[0, 3] = PLAYER1
    board2[0, 4] = PLAYER1
    board2[1, 1] = PLAYER2
    board2[1, 2] = PLAYER1
    board2[1, 3] = PLAYER2
    board2[1, 4] = PLAYER2
    board2[2, 2] = PLAYER2
    board2[2, 3] = PLAYER1
    board2[2, 4] = PLAYER1
    board2[3, 2] = PLAYER1
    board2[3, 3] = PLAYER1

    board3 = np.zeros((6, 7), BoardPiece)       #possible played board
    board3[0, 1] = PLAYER2
    board3[0, 2] = PLAYER2
    board3[0, 3] = PLAYER2

    board = np.zeros((6, 7), BoardPiece)
    board[0, 1] = BoardPiece(2)
    board[0, 2] = BoardPiece(2)
    board[0, 3] = BoardPiece(1)
    board[0, 4] = BoardPiece(1)
    board[1, 1] = BoardPiece(2)
    board[1, 2] = BoardPiece(1)
    board[1, 3] = BoardPiece(2)
    board[1, 4] = BoardPiece(2)
    board[2, 2] = BoardPiece(2)
    board[2, 3] = BoardPiece(1)
    board[2, 4] = BoardPiece(1)
    board[3, 2] = BoardPiece(1)
    board[3, 3] = BoardPiece(1)
    board[0, 5] = BoardPiece(1)
    #board[0, 6] = BoardPiece(1)
    board[3, 4] = BoardPiece(1)
    #board[4, 5] = BoardPiece(1)
    board[5, 6] = BoardPiece(2)
    board[5, 5] = BoardPiece(2)
    board[5, 4] = BoardPiece(2)
    board[5, 3] = BoardPiece(2)

    board_l = np.zeros((6, 7), BoardPiece)      # board with left diagonal for Player 2
    board_l[0, 3] = PLAYER2
    board_l[1, 2] = PLAYER2
    board_l[2, 1] = PLAYER2
    board_l[3, 0] = PLAYER2

    board_6421 = np.zeros((6, 7), BoardPiece)
    board_6421[0, 6] = PLAYER1
    board_6421[0, 4] = PLAYER2
    board_6421[0, 2] = PLAYER1
    # board_6421[0, 1] = PLAYER2

    board_1420 = np.zeros((6, 7), BoardPiece)
    board_1420[0, 1] = PLAYER2
    board_1420[0, 4] = PLAYER1
    board_1420[0, 2] = PLAYER2
    # board_1420[0, 0] = PLAYER1

    def test_one(self):
        pass

    def test_buildGameStateFromID(self):
        node = GameState(PLAYER1, '6421', self.board0)
        node2 = GameState(PLAYER2, '1420', self.board0)
        node.buildGameStateFromID()
        print(pretty_print_board(node.gameState))
        print(pretty_print_board(self.board_6421))
        assert (node.status is None) and (node.gameState == self.board_6421).all()
        node2.buildGameStateFromID()
        print(pretty_print_board(node2.gameState))
        print(pretty_print_board(self.board_1420))
        assert (node2.status is None) and (node2.gameState == self.board_1420).all()

        node3 = GameState(PLAYER2, '1420', self.board1)
        node3.buildGameStateFromID()
        print(pretty_print_board(node3.gameState))
        print(pretty_print_board(self.board1))
        assert node3.status == 'impossible'
