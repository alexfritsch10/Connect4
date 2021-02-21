import copy

from common import GameState, get_available_moves
from player import Player

class GameController:

    def __init__(self, game, playerOne: Player, playerTwo: Player):
        self.game = game
        self.playerOne = playerOne
        self.playerTwo = playerTwo
        self.trainingHistory = []

    def playManyGames(self, numberOfGames: int):
        playerOneWins = 0
        playerTwoWins = 0
        draws = 0
        for i in range(numberOfGames):
            self.game.resetBoard()
            self.playGame()
            gameState = self.game.getGameState()
            print("Game State: " + str(gameState))
            if gameState == 1:
                playerOneWins += 1
            elif gameState == 2:
                playerTwoWins += 1
            else:
                draws = draws + 1
        totalWins = playerOneWins + playerTwoWins + draws
        print('One Wins: ' + str(int(playerOneWins * 100 / totalWins)) + '%')
        print('Two Wins: ' + str(int(playerTwoWins * 100 / totalWins)) + '%')
        print('Draws: ' + str(int(draws * 100 / totalWins)) + '%')
        print('Draws: ' + str(draws))

    def playGame(self):
        print("In play Game")
        playerToMove = self.playerOne
        while self.game.getGameState() == -1:
            availableMoves = get_available_moves(self.game.getBoard())
            #print("Available Moves: ")
            #print(availableMoves)
            move = playerToMove.getMove(availableMoves, self.game.getBoard())
            #print("Going to execute move: " + str(move))
            self.game.move(move, playerToMove)
            playerToMove = self.switchPlayer(playerToMove)

        #print("Board History")
        #print(self.game.getBoardHistory())
        for historyItem in self.game.getBoardHistory():
            self.trainingHistory.append((self.game.getGameState(), copy.deepcopy(historyItem)))

    def getTrainingHistory(self):
        return self.trainingHistory

    def switchPlayer(self, playerToMove: Player) -> Player:
        if playerToMove == self.playerOne:
            return self.playerTwo
        return self.playerOne
