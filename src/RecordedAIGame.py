import numpy as np
from keras.models import load_model
from src.Player import Player
from src.Game import Game
from src.Board import Board


class RecordedAIGame(Game):
    def __init__(self, board, player_one, player_two):
        Game.__init__(self, board, player_one, player_two)
        self.history = dict()

    def initialize(self):
        self.history[1] = {'moves': list(), 'states': list()}
        self.history[2] = {'moves': list(), 'states': list()}

    def play_a_game(self):
        self.board.playing = True
        while self.board.playing and self.board.plays < 42:
            self.history[1]['states'].append(self.board.board)
            p1move, win = self.player_one.best_move()
            self.history[1]['moves'].append(p1move.col)
            print(self.board)
            if win:
                self.winner = 1
                self.board.playing = False
            elif self.board.plays < 42:
                self.history[2]['states'].append(self.uniform_board())
                p2move, win = self.player_two.play()
                #p2move, win = self.player_two.best_move()
                self.history[2]['moves'].append(p2move.col)
                print(self.board)
                if win:
                    self.winner = 2
                    self.board.playing = False
        if self.board.plays == 42:
            self.winner = None

    def export_history(self, filename):
        if self.winner is not None:
            np.save(f'{filename}_states.npy',
                    np.array(self.history[self.winner]['states']))
            np.save(f'{filename}_moves.npy',
                    np.array(self.history[self.winner]['moves']))

    def uniform_board(self):
        new_board = []
        for j in self.board.board:
            row = []
            for i in j:
                row.append(int(1 / (0.5 * i) if i else i))
            new_board.append(row)
        return new_board

    def export_final_results(self):
        if self.winner is not None:
            moves = np.array(self.history[self.winner]['moves'])
            states = np.array(self.history[self.winner]['states'])
            return moves, states, 'ok'
        return None, None, None


if __name__ == '__main__':
    b = Board()
    model = load_model('my_first_model.h5')
    p1 = Player(1, 'red', b, model, 0.1)
    p2 = Player(2, 'yellow', b, model, 0.1)
    g = RecordedAIGame(b, p1, p2)
    g.initialize()
    g.play_a_game()
    print(g.export_final_results())
