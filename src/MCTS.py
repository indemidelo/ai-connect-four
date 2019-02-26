from copy import deepcopy
from src.NNPlayer import NNPlayer
from src.Game import Game


class MonteCarloTreeSearch():
    def __init__(self, board, nn_player: NNPlayer, n_iter):
        self.board = board
        self.player = nn_player
        self.n_iter = n_iter
        self.rollout_pol = {j: 0. for j in range(7)}
        self.wins = 0

    def tree_search(self):
        available_moves = self.board.list_available_moves()
        for col in available_moves:
            for j in range(self.n_iter):
                new_b = deepcopy(self.board)
                p1dummy = NNPlayer(
                    self.player.name, new_b, self.player.model, True)
                opponent = 1 if self.player.name == 2 else 2
                p2dummy = NNPlayer(
                    opponent, new_b, self.player.model, True)
                g = Game(new_b, p1dummy, p2dummy)
                g.play_a_game(first_move=col)
                n_new_plays = new_b.plays - self.board.plays
                if g.winner == self.player.name:
                    self.rollout_pol[col] += 0.9 ** n_new_plays
                    self.wins += 1
                elif g.winner is None:
                    self.rollout_pol[col] += 0.5 * 0.9 ** n_new_plays
                else:
                    self.rollout_pol[col] -= 0.9 ** (n_new_plays - 1)
        return self.rollout_pol
