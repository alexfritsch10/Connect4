from common import check_end_state, initialize_game_state, PLAYER1, PLAYER2, GameState, apply_player_action, PlayerAction
from player import Player

class Game:

    def __init__(self):
        self.resetBoard()

    def resetBoard(self):
        self.board = initialize_game_state()

    def getGameState(self):
        checkPlayerOne = check_end_state(self.board, PLAYER1)
        print("Check Player One: " + str(checkPlayerOne))
        checkPlayerTwo = check_end_state(self.board, PLAYER2)
        print("Check Player Two: " + str(checkPlayerTwo))
        if checkPlayerOne == GameState.IS_WIN:
            return 1
        elif checkPlayerOne == GameState.IS_DRAW and checkPlayerTwo == GameState.IS_DRAW:
            return 0
        elif checkPlayerOne == GameState.STILL_PLAYING and not checkPlayerTwo == GameState.IS_WIN:
            return -1
        elif checkPlayerTwo == GameState.IS_WIN:
            return 2
        else:
            # Error Handling
            return -2

    def move(self, move: PlayerAction, player: Player):
        apply_player_action(self.board, move, player.getNumber())

    def getBoard(self):
        return self.board