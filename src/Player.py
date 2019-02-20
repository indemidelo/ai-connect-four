import random
import numpy as np
from src.Move import Move


class Player():
    def __init__(self, name, board, model=None, qualcosa=None):
        self.name = name
        self.board = board
        self.model = model
        self.qualcosa = qualcosa

    def play(self, fixed_move=None):
        col = self.fixed_move(fixed_move)
        if col != -1:
            m = Move(self.name, self.board, col)
            m.play()
            win = self.board.check_connect(m)
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
            col = random.choice(available_moves)
            return col
        else:
            return -1

    def fixed_move(self, fixed_move):
        if fixed_move == None:
            return self.random_move()
        else:
            return fixed_move

    def best_move(self):
        if random.random() > self.qualcosa:
            available_moves = self.search_available_moves()
            rebased_board = self.board.board.reshape((1, 6, 7, 1))
            probs = self.model.predict(rebased_board)
            print(f'Player {self.name} had these choices: {probs}')
            col = np.argmax(probs)
            if col in available_moves:
                return self.play(col)
            else:
                return self.play()
        else:
            return self.play()

