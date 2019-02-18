import random
from src.Move import Move


class Player():
    def __init__(self, name, color, board):
        self.name = name
        self.color = color
        self.board = board

    def play(self, fixed_move=None):
        col = self.fixed_move(fixed_move)
        if col != -1:
            m = Move(self.name, self.board, col)
            m.play()
            # print(f'Player {self.name} played column {col}')
            win = self.board.check_connect(m)
            # if result: print(result)
            return m, win
        else:
            self.board.playing = False
            self.board.full = True

    def search_available_moves(self):
        av_moves = list()
        for j in range(self.board.board.shape[1]):
            if 0 in self.board.board[:, j]:
                av_moves.append(j)
        return av_moves

    def random_move(self):
        available_moves = self.search_available_moves()
        if available_moves:
            return random.choice(available_moves)
        else:
            return -1

    def fixed_move(self, fixed_move):
        if fixed_move == None:
            return self.random_move()
        else:
            return fixed_move
