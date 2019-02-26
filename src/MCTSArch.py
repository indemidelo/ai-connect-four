from multiprocessing import Queue
from src.SingleMCTS import SingleMonteCarloTreeSearch


class MonteCarloTreeSearchArch():
    def __init__(self, mcts_iter: int):
        self.mcts_iter = mcts_iter
        self.queues_in = [Queue()] * 7
        self.queues_out = [Queue()] * 7
        self.processes = list()

    def initialize(self):
        for j in range(7):
            p = SingleMonteCarloTreeSearch(
                j, self.mcts_iter, self.queues_in[j], self.queues_out[j])
            p.start()
            self.processes.append(p)

    def tree_search(self, board, player):
        available_moves = board.list_available_moves()
        rollout_policy = [0.0] * 7
        for col in available_moves:
            self.queues_in[col].put((board, player, col))
        for col in available_moves:
            rollout_policy[col], wins = self.queues_out[col].get()
        return rollout_policy
