from game import Game
from player import Player
from gameController import GameController

if __name__ == "__main__":
    # simulates an arbitrary number of games between a random player and a player using our model
    game = Game()
    redPlayer = Player(1, strategy='random')
    yellowPlayer = Player(2, strategy='model')

    gameController = GameController(game, redPlayer, yellowPlayer)
    print("Playing with one random player and one neural player")
    gameController.playManyGames(100)
