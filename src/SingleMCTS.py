from copy import deepcopy
from threading import Thread
from queue import Queue
from src.NNPlayer import NNPlayer
from src.Game import Game


class SingleMonteCarloTreeSearch(Thread):
    def __init__(self, col: int, n_iter: int,
                 queue_in: Queue, queue_out: Queue):
        Thread.__init__(self)
        self.col = col
        self.n_iter = n_iter
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):
        while 1:
            board, player, col = self.queue_in.get()
            results = self.single_tree_search(player, board)
            self.queue_out.put(results)

    def single_tree_search(self, player, board):
        rollout_policy, wins = 0, 0
        for j in range(self.n_iter):
            new_b = deepcopy(board)
            p1dummy = NNPlayer(
                player.name, new_b, player.model, True)
            opponent = 1 if player.name == 2 else 2
            p2dummy = NNPlayer(
                opponent, new_b, player.model, True)
            g = Game(new_b, p1dummy, p2dummy)
            g.play_a_game(first_move=self.col)
            n_new_plays = new_b.plays - board.plays
            if g.winner == player.name:
                rollout_policy += 0.9 ** n_new_plays
                wins += 1
            elif g.winner is None:
                rollout_policy += 0.5 * 0.9 ** n_new_plays
            else:
                rollout_policy -= 0.9 ** (n_new_plays - 1)
        return rollout_policy, wins
