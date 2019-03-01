from copy import deepcopy
from threading import Thread
from queue import Queue
from src.Player import Player
from src.tfPlayer import tfPlayer
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
            board, player = self.queue_in.get()
            results = self.single_tree_search(player, board)
            self.queue_out.put(results)

    def single_tree_search(self, player, board):
        reward, wins = 0, 0
        for j in range(self.n_iter):
            winner, n_new_plays = self.rollout_game(player, board)
            if winner == player.name:
                reward += 0.9 ** n_new_plays
                wins += 1
            elif winner is None:
                reward += 0.5 * 0.9 ** n_new_plays
            else:
                reward -= 0.9 ** (n_new_plays - 1)
        return reward, wins

    def rollout_game(self, player, board):
        new_b = deepcopy(board)
        p1dummy = tfPlayer(
            player.name, new_b,
            player.sess, player.pred,
            player.input_placeholder, True)
        opponent = 1 if player.name == 2 else 2
        p2dummy = Player(opponent, new_b)
        g = Game(new_b, p1dummy, p2dummy)
        g.play_a_game(first_move=self.col)
        n_new_plays = new_b.plays - board.plays
        return g.winner, n_new_plays