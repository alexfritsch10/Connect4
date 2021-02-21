import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, pretty_print_board, column_to_be_played_for_win
import pytest


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

    board4 = np.zeros((6, 7), BoardPiece)  # possible played board
    board4[0, 1] = PLAYER2
    board4[0, 2] = PLAYER1
    board4[1, 1] = PLAYER2
    board4[2, 1] = PLAYER2
    board4[0, 3] = PLAYER2

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

    def test_one(self):
        pass

    def test_initialize_game_state(self):
        from agents.common import initialize_game_state

        ret = initialize_game_state()

        assert isinstance(ret, np.ndarray)
        assert ret.dtype == BoardPiece
        assert ret.shape == (6, 7)
        assert np.all(ret == NO_PLAYER)

    def test_pretty_print_board(self):
        from agents.common import pretty_print_board

        pretty_board = pretty_print_board(self.board2)
        assert isinstance(pretty_board, str)

        print(pretty_board)
        #assert(pretty_board == "|==============|\n|              |\n|              |\n|    X X       |\n|    O X X     |\n|  O X O O     |\n|  O O X X     |\n|==============|\n|0 1 2 3 4 5 6 |")
        assert True

    def test_apply_player_action(self):
        from agents.common import apply_player_action, pretty_print_board

        action = 5
        board0 = apply_player_action(self.board0.copy(), action, BoardPiece(1), False)

        #with pytest.raises(ValueError):
        #    apply_player_action(self.board1, 3, BoardPiece(2), True)

        print(pretty_print_board(board0))
        assert board0[0, action] == 1

        board0 = apply_player_action(board0, action, BoardPiece(1), False)
        print(pretty_print_board(board0))
        assert board0[1, action] == 1
        board0 = apply_player_action(board0, action, BoardPiece(1), False)
        print(pretty_print_board(board0))
        assert board0[2, action] == 1
        board0 = apply_player_action(board0, action, BoardPiece(1), False)
        print(pretty_print_board(board0))
        assert board0[3, action] == 1
        board0 = apply_player_action(board0, action, BoardPiece(1), False)
        print(pretty_print_board(board0))
        assert board0[4, action] == 1
        board0 = apply_player_action(board0, action, BoardPiece(1), False)
        print(pretty_print_board(board0))
        assert board0[5, action] == 1

        # with pytest.raises(ValueError):
        #     board2 = apply_player_action(board0, action, BoardPiece(1), False)
        #    print(pretty_print_board(board2))


    def test_connected_four(self):
        from agents.common import connected_four, pretty_print_board

        print(pretty_print_board(self.board))
        player1_won = connected_four(self.board, BoardPiece(1))
        player2_won = connected_four(self.board, BoardPiece(2))
        assert not player1_won
        assert player2_won
        print(pretty_print_board(self.board_l))
        player2_won = connected_four(self.board_l, PLAYER2)
        assert player2_won

    def test_check_end_state(self):
        from agents.common import check_end_state, GameState, pretty_print_board

        print(pretty_print_board(self.board1))
        print("expected: player 1 won")
        assert check_end_state(self.board1, 1) == GameState.IS_WIN
        print("expected: player 2 draw")
        assert check_end_state(self.board1, 2) == GameState.IS_DRAW
        print(pretty_print_board(self.board0))
        print("expected: still playing")
        assert check_end_state(self.board0, 1) == GameState.STILL_PLAYING

    def test_column_to_be_played_for_win(self):

        print(pretty_print_board(self.board3))
        assert 0 == column_to_be_played_for_win(self.board3, PLAYER2)
        self.board3[0, 0] = PLAYER1
        print(pretty_print_board(self.board3))
        assert 4 == column_to_be_played_for_win(self.board3, PLAYER2)
        assert -1 == column_to_be_played_for_win(self.board3, PLAYER1)

        print(pretty_print_board(self.board4))
        assert 1 == column_to_be_played_for_win(self.board4, PLAYER2)
        self.board4[3, 1] = PLAYER1
        print(pretty_print_board(self.board4))
        assert -1 == column_to_be_played_for_win(self.board4, PLAYER2)
        assert -1 == column_to_be_played_for_win(self.board4, PLAYER1)

    def test_generate_move_random(self):
        from agents.agent_random.random import generate_move_random

        action = generate_move_random(self.board0, PLAYER2, None)
        print(action)
        assert True

    def test_evaluate_heuristic(self):
        from agents.agent_minimax.minimax import evaluate_heuristic

        print(pretty_print_board(self.board2))
        heuristic_c5 = evaluate_heuristic(self.board2, 5, PLAYER1)
        heuristic_c0 = evaluate_heuristic(self.board2, 0, PLAYER1)
        heuristic_c1 = evaluate_heuristic(self.board2, 1, PLAYER1)
        heuristic_c2 = evaluate_heuristic(self.board2, 2, PLAYER1)
        heuristic_c3 = evaluate_heuristic(self.board2, 3, PLAYER1)
        heuristic_c4 = evaluate_heuristic(self.board2, 4, PLAYER1)
        heuristic_c6 = evaluate_heuristic(self.board2, 6, PLAYER1)

        print("solution column 0: ", heuristic_c0)
        print("solution column 1: ", heuristic_c1)
        print("solution column 2: ", heuristic_c2)
        print("solution column 3: ", heuristic_c3)
        print("solution column 4: ", heuristic_c4)
        print("solution column 5: ", heuristic_c5)
        print("solution column 6: ", heuristic_c6)

        assert heuristic_c5 == 4
