import random
from src.Move import Move
from src.Board import Board


class NNPlayer():
    def __init__(self, name: int, board: Board,
                 model, training: bool=False):
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
        #nn_policy = self.model(self.board.board_as_tensor(self.name))
        #print('board as tensor', self.board.board_as_tensor(self.name))
        nn_policy = self.model.predict(
            self.board.board_as_tensor(self.name), verbose=False).flatten()
        nn_policy = {j: max(0, v) for j, v in enumerate(nn_policy)}
        sum_policy = sum(nn_policy.values()) or 1.0
        nn_policy = {j: v/sum_policy for j, v in nn_policy.items()}
        #print(f'Player {self.name} policy: {nn_policy}')
        if not available_moves:
            return -1
        weight_ko_moves = sum([v for k, v in enumerate(nn_policy) if
                               k not in available_moves])
        nn_policy_ok = dict()
        for k, v in nn_policy.items():
            nn_policy_ok[k] = v + weight_ko_moves \
                if k in available_moves else 0.0
        if self.training:
            return random.choices(list(nn_policy_ok.keys()),
                                  list(nn_policy_ok.values()))[0]
        else:
            best_move = sorted(nn_policy_ok.items(), key=lambda x: x[1])[-1][0]
            return best_move

