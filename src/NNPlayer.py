import random
from src.Move import Move
from src.Board import Board


class NNPlayer():
    def __init__(self, name: int, board: Board,
                 model, training: bool = False):
        self.name = name
        self.board = board
        self.model = model
        self.training = training

    def play(self, fixed_move=None):
        if fixed_move is None:
            col = self.nn_move()
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

    def nn_move(self):
        available_moves = self.board.list_available_moves()
        if not available_moves:
            return -1
        nn_policy = self.model.predict(
            self.board.board_as_tensor(self.name), verbose=False).flatten()
        min_nnp = abs(min(nn_policy))
        nn_policy = {j: v + min_nnp for j, v in enumerate(nn_policy)}
        sum_policy = sum(nn_policy.values()) or 1.0
        nn_policy = {j: v / sum_policy for j, v in nn_policy.items()}
        sum_unavail_moves = sum(
            [v for k, v in enumerate(nn_policy) if k not in available_moves])
        nn_policy = {k: v + sum_unavail_moves if k in available_moves
                     else 0.0 for k, v in nn_policy.items()}
        if self.training:
            if sum(nn_policy.values()) > 0.0:
                return random.choices(
                    list(nn_policy.keys()), list(nn_policy.values()))[0]
            else:
                return random.choice(nn_policy.keys())
        else:
            best_move = sorted(nn_policy.items(), key=lambda x: x[1])[-1][0]
            return best_move
