#import torch
import time
from queue import Queue
from src.MCTSArch import MonteCarloTreeSearchArch


class NNRecordedGame_mp():
    def __init__(self, board, player_one, player_two, mcts_iter):
        """
        A game of n-connect
        :param board:
        :param player_one:
        :param player_two:
        """
        self.board = board
        self.player_one = player_one
        self.player_two = player_two
        self.winner = None
        self.board.playing = True
        self.mcts = MonteCarloTreeSearchArch(mcts_iter)
        self.queues_in = [Queue() for _ in range(7)]
        self.queues_out = [Queue() for _ in range(7)]
        self.processes = list()
        self.history = dict()

    def initialize(self):
        self.history[self.player_one.name] = {
            'moves': list(), 'states': list(), 'rollout_pol': list()}
        self.history[self.player_two.name] = {
            'moves': list(), 'states': list(), 'rollout_pol': list()}
        self.mcts.initialize()

    def play_a_game(self, print_board=False, first_move=None):
        """ To play a game """
        while self.board.playing and not self.board.full:
            if first_move is not None:
                p1move, win = self.player_one.play(first_move)
                first_move = None
            else:
                win = self.turn(self.player_one)
                if print_board:
                    print(f'Player {self.player_one.name} move')
                    print(self.board)
            if win:
                self.winner = self.player_one.name
                self.board.playing = False
            elif not self.board.full:
                win = self.turn(self.player_two)
                if print_board:
                    print(f'Player {self.player_two.name} move')
                    print(self.board)
                if win:
                    self.winner = self.player_two.name
                    self.board.playing = False
        if print_board:
            print(self.board)

    def turn(self, player):
        state = self.board.board_as_tensor(player.name)
        #state = state.to(self.device, dtype=torch.float64)
        self.history[player.name]['states'].append(state)
        start_t = time.time()
        rollout_policy = self.mcts.tree_search(self.board, player)
        print('ELapsed: ', time.time() - start_t)
        self.history[player.name]['rollout_pol'].append(rollout_policy)
        move, win = player.play()
        #self.history[player.name]['moves'].append(move.col)
        return win
