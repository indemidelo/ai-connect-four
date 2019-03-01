#import torch
import time
import numpy as np
from src.MCTS import MonteCarloTreeSearch


class NNRecordedGame():
    def __init__(self, board, player_one, player_two, n_iter):
        """
        A game of n-connect
        :param board:
        :param player_one:
        :param player_two:
        """
        self.board = board
        self.player_one = player_one
        self.player_two = player_two
        self.n_iter = n_iter
        self.winner = None
        self.board.playing = True
        self.history = dict()

    def initialize(self):
        self.history[self.player_one.name] = {
            'states': [], 'rollout_pol': []}
        self.history[self.player_two.name] = {
            'states': [], 'rollout_pol': []}

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
        self.history[player.name]['states'].append(state)
        mcts = MonteCarloTreeSearch(self.board, player, self.n_iter)
        start_time = time.time()
        rollout_policy = mcts.tree_search()
        print('Elapsed: ', time.time()-start_time)
        self.history[player.name]['rollout_pol'].append(rollout_policy)
        move, win = player.play()
        return win
