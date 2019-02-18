import numpy as np
from src.Player import Player
from src.Game import Game
from src.Board import Board


class RecordedGame(Game):
    def __init__(self, board, player_one, player_two):
        Game.__init__(self, board, player_one, player_two)
        self.history = dict()

    def initialize(self):
        self.history[1] = {'moves': list(), 'states': list()}
        self.history[2] = {'moves': list(), 'states': list()}

    def play_a_game(self):
        self.board.playing = True
        while self.board.playing:
            self.history[1]['states'].append(self.board.board)
            p1move = self.player_one.play()
            self.history[1]['moves'].append(p1move.col)
            if self.board.playing:
                self.history[2]['states'].append(self.uniform_board())
                p2move = self.player_two.play()
                self.history[2]['moves'].append(p2move.col)
            elif not self.board.full:
                self.winner = self.player_one.name
            print(self.board)
        if not self.winner and not self.board.full:
            self.winner = self.player_two.name
        elif not self.winner and self.board.full:
            self.winner = 'TIE'
            print('The game is a tie')

    def export_history(self, filename):
        if self.winner is not 'TIE':
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
        if self.winner is not 'TIE':
            moves = np.array(self.history[self.winner]['moves'])
            states = np.array(self.history[self.winner]['states'])
            return moves, states


if __name__ == '__main__':
    b = Board()
    p1 = Player(1, 'red', b)
    p2 = Player(2, 'yellow', b)
    g = RecordedGame(b, p1, p2)
    g.initialize()
    g.play_a_game()
    print(g.export_final_results())
