import random
from src.Move import Move


class Player():
    def __init__(self, name, color, board):
        self.name = name
        self.color = color
        self.board = board

    def play(self, fixed_move=None):
        col = self.fixed_move(fixed_move)
        m = Move(self.name, self.board, col)
        m.play()
        result = self.board.check_connect(m)
        print(result)

    def random_move(self):
        col = random.choice(range(0, 7))
        return col

    def fixed_move(self, fixed_move):
        if fixed_move == None:
            return self.random_move()
        else:
            return fixed_move

    def check_connect(self, board, last_move):
        return
