import random
from copy import deepcopy
from src.Move import Move
from src.Game import Game
from src.Player import Player
from src.OneMoveSmartPlayer import OneMoveSmartPlayer


class RLPlayer():
    def __init__(self, name, board, n_iter):
        self.name = name
        self.board = board
        self.n_iter = n_iter

    def play(self, fixed_move=None):
        if fixed_move is None:
            col = self.rlmove()
        else:
            col = fixed_move
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

    def rlmove(self):
        available_moves = self.search_available_moves()
        if available_moves:
            col = self.mcts(available_moves)
            return col
        else:
            return -1

    def mcts(self, moves):
        results = {col: 0 for col in moves}
        for col in moves:
            for j in range(self.n_iter):
                new_b = deepcopy(self.board)
                p1dummy = Player(self.name, new_b, 1)
                opponent = 1 if self.name == 2 else 2
                #p2dummy = OneMoveSmartPlayer(opponent, new_b)
                p2dummy = Player(opponent, new_b)
                g = Game(new_b, p1dummy, p2dummy)
                g.play_a_game(first_move=col)
                n_new_plays = new_b.plays - self.board.plays
                #print(f'{n_new_plays} - {g.winner} %%', end=' ')
                if g.winner == self.name:
                    results[col] += 0.9**n_new_plays
                elif g.winner is None:
                    results[col] += (0.5*0.9)**n_new_plays
                else:
                    results[col] -= 0.9**(n_new_plays - 1)
        print(f'Player {self.name} results: {results}')
        return max(results.items(), key=lambda x: x[1])[0]

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
