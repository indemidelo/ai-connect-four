from queue import Queue
from copy import deepcopy
from src.SingleMCTS import SingleMonteCarloTreeSearch


class MonteCarloTreeSearchArch():
    def __init__(self, mcts_iter: int):
        self.mcts_iter = mcts_iter
        self.queues_in = [Queue()] * 7
        self.queues_out = [Queue()] * 7
        self.processes = list()
        self.rewards = {j: 0.0 for j in range(7)}

    def initialize(self):
        for j in range(7):
            p = SingleMonteCarloTreeSearch(
                j, self.mcts_iter, self.queues_in[j], self.queues_out[j])
            p.start()
            self.processes.append(p)

    def tree_search(self, board, player):
        available_moves = board.list_available_moves()
        for col in available_moves:
            self.queues_in[col].put((deepcopy(board), player))
        for col in available_moves:
            if not self.queues_out[col].empty():
                self.rewards[col], wins = self.queues_out[col].get()
        policy = self.rewards_to_policy()
        return policy

    def rewards_to_policy(self):
        min_rp = abs(min(self.rewards.values()))
        rollout_policy = {k: v + min_rp for k, v in self.rewards.items()}
        sum_policy = sum(rollout_policy.values()) or 1.0
        rollout_policy = [v / sum_policy if k in self.rewards else 0.0
                          for k, v in rollout_policy.items()]
        return rollout_policy