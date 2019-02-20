import random
import numpy as np
from src.Game import Game
from src.RLPlayer import RLPlayer
from src.Player import Player
from src.Board import Board


class RLGameBatch():
    def __init__(self, player_one, player_two, n_games):
        self.player_one = player_one
        self.player_two = player_two
        self.n_games = n_games
        self.states = np.empty((0, 6, 7))
        self.actions = np.empty((0,))
        self.results = {self.player_one.name: 0,
                        self.player_two.name: 0}

    def fire(self, lograte, print_board=False):
        for j in range(self.n_games):
            if j % lograte == 0:
                print(f'Playing Game {j + 1}')
            b = Board()
            self.player_one.board = b
            self.player_two.board = b
            g = Game(b, self.player_one, self.player_two)
            g.play_a_game(print_board)
            if j % lograte == 0:
                print(f'Player {g.winner} won')
            if g.winner is not None:
                self.results[g.winner] += 1
        print(f'Player {self.player_one.name} won {self.results[self.player_one.name]} time'
              f's\nPlayer {self.player_two.name} won {self.results[self.player_two.name]} tim'
              f'es')


if __name__ == '__main__':
    random.seed(1)
    b = Board()
    n_rollout = 500
    p1 = RLPlayer(1, b, n_rollout)
    p2 = RLPlayer(2, b, n_rollout)
    n_games = 1
    batch = RLGameBatch(p1, p2, n_games)
    batch.fire(1, print_board=True)
