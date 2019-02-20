import random
from copy import deepcopy
from src.Move import Move
from src.Game import Game
from src.Player import Player


class OneMoveSmartPlayer():
    def __init__(self, name, board):
        self.name = name
        self.board = board

    def play(self, fixed_move=None):
        if fixed_move is None:
            col = self.smart_move()
        else:
            col = fixed_move
        if col != -1:
            m = Move(self.name, self.board, col)
            m.play()
            win = self.board.check_connect(m)
            return m, win
        else:
            self.board.playing = False
            self.board.full = True

    def smart_move(self):
        available_moves = self.search_available_moves()
        if available_moves:
            for col in available_moves:
                new_b = deepcopy(self.board)
                p1dummy = Player(self.name, new_b)
                opponent = 1 if self.name == 2 else 2
                p2dummy = Player(opponent, new_b)
                g = Game(new_b, p1dummy, p2dummy)
                g.one_move(col)
                if g.winner == self.name:
                    return col
            return random.choice(available_moves)
        else:
            return -1

    def search_available_moves(self):
        av_moves = list()
        for j in range(self.board.board.shape[1]):
            if 0 in self.board.board[:, j]:
                av_moves.append(j)
        return av_moves
