from copy import deepcopy
from src.tfPlayer import tfPlayer
from src.Player import Player
from src.Game import Game


class MonteCarloTreeSearch():
    def __init__(self, board, tf_player: tfPlayer, n_iter):
        self.board = board
        self.player = tf_player
        self.n_iter = n_iter
        self.rewards = {j: 0.0 for j in range(7)}
        self.wins = 0

    def tree_search(self):
        available_moves = self.board.list_available_moves()
        for col in available_moves:
            for j in range(self.n_iter):
                winner, n_new_plays = self.rollout_game(col)
                if winner == self.player.name:
                    self.rewards[col] += 0.9 ** n_new_plays
                    self.wins += 1
                elif winner is None:
                    self.rewards[col] += 0.5 * 0.9 ** n_new_plays
                else:
                    self.rewards[col] -= 0.9 ** (n_new_plays - 1)
        return self.rewards_to_policy()

    def rewards_to_policy(self):
        min_rp = abs(min(self.rewards.values()))
        rollout_policy = {k: v + min_rp for k, v in self.rewards.items()}
        sum_policy = sum(rollout_policy.values()) or 1.0
        rollout_policy = [v / sum_policy if k in self.rewards else 0.0
                          for k, v in rollout_policy.items()]
        return rollout_policy

    def rollout_game(self, col):
        new_b = deepcopy(self.board)
        p1dummy = tfPlayer(
            self.player.name, new_b,
            self.player.sess, self.player.pred,
            self.player.input_placeholder, True)
        opponent = 1 if self.player.name == 2 else 2
        p2dummy = Player(opponent, new_b)
        g = Game(new_b, p1dummy, p2dummy)
        g.play_a_game(first_move=col)
        n_new_plays = new_b.plays - self.board.plays
        return g.winner, n_new_plays
