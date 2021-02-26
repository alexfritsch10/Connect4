from game import Game
from player import Player
from gameController import GameController
from model import Model

if __name__ == "__main__":
    firstGame = Game()
    redPlayer = Player(1, strategy='random')
    yellowPlayer = Player(2, strategy='random')

    gameController = GameController(firstGame, redPlayer, yellowPlayer)
    print("Playing with both players with random strategies")
    gameController.playManyGames(100)
    print("Print in main.py")
    print(gameController.getTrainingHistory()[:10])

    model = Model()
    model.train(gameController.getTrainingHistory())

    #redNeuralPlayer = Player(1, strategy='model', model=model)
    #yellowNeuralPlayer = Player(2, strategy='model', model=model)

    #secondGame = Game()
    #gameController = GameController(secondGame, redPlayer, yellowNeuralPlayer)
    #print ("Playing with yellow player as Neural Network")
    #gameController.playManyGames(1000)

    #thirdGame = Game()
    #gameController = GameController(thirdGame, redNeuralPlayer, yellowPlayer)
    #print("Playing with red player as Neural Network")
    #gameController.playManyGames(1000)
