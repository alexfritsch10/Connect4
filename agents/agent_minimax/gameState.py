import numpy as np

from agents.agent_minimax.heuristic import evaluate_heuristic
from agents.common import BoardPiece, apply_player_action, connected_four, column_to_be_played_for_win, other_player, \
    move_is_possible, pretty_print_board


class GameState:
    def __init__(self, player: BoardPiece, positionID: str, board: np.ndarray):   # vorher: = None
        """
        recursively building game tree, every middle node (those who are less than depth 4) has list of children,
        building when GameState with root is called. All nodes on depth 4 get a board copy, it'll get modified later by
        buildGameStateFromID and the value of all nodes is None at the beginning.
        """
        self.positionID = positionID
        self.player = player
        self.status = None                 # added self status in case a combination of moves is impossible
        self.value = None
        #if positionID == '':
        #    self.bmove = None
        # if board is not None:
        #    self.gameState = board.copy()  # use board.copy() anstatt board, sonst machst du alle changes auf dem gleichen board
        if len(positionID) < 4:
            # print('positionID <4:', positionID)
            self.children = []
            for i in range(7):
                self.children.append(GameState(other_player(player), positionID + str(i), board))
        else:
            # print('positionID >= 4: ', positionID)
            self.board = board.copy()
            #if positionID == '6666':
            #    print('board: ', pretty_print_board(self.board))

        # besser: if positionID.len < 4:
        #           children
        #         else:
        #           self.board = board

  #  def computeScore(self):
  #      if connected_four(self.gameState, self.player):
  #          return 100
  #      elif column_to_be_played_for_win(self.gameState, self.player) != -1:
  #          return -100
  #      else:
  #          return 0

    def set_value(self):
        """
        uses heuristic to set value of node,
        last character of positionID is the move that should be evaluated by heuristic
        """
        if move_is_possible(self.board, int(self.positionID[-1])):
            self.value = evaluate_heuristic(self.board, int(self.positionID[-1]), self.player)
        else:
            self.status = 'impossible'

    def buildGameStateFromID(self):
        """
        plays 3 moves on self.board if they're possible
        """
        # positionID = eg 4256 -> 1. Move: 4th Column, 2. Move: 2nd column
        player = self.player
        for col in self.positionID[:-1]:                                    # try: only make the first 3 moves, before: complete string
            if move_is_possible(self.board, int(col)):                            # only apply possible moves
                apply_player_action(self.board, int(col), player)
                player = other_player(player)
            else:
                self.status = 'impossible'
                break
        # hier hast du den player am anfang auf 1 gesetzt, egal ob player 1 oder player 2 spielt





