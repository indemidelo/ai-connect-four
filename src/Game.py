from src.Board import Board
from src.Player import Player


class Game():
    def __init__(self, board, player_one, player_two):
        self.board = board
        self.player_one = player_one
        self.player_two = player_two
        self.winner = None

    def play_a_game(self, print_board=False, first_move=None):
        self.board.playing = True
        while self.board.playing and not self.board.full:
            if first_move is not None:
                p1move, win = self.player_one.play(first_move)
                first_move = None
            else:
                p1move, win = self.player_one.play()
            if print_board:
                print(self.board)
            if win:
                self.winner = 1
                self.board.playing = False
            elif not self.board.full:
                p2move, win = self.player_two.play()
                if print_board:
                    print(self.board)
                if win:
                    self.winner = 2
                    self.board.playing = False


if __name__ == '__main__':
    b = Board()
    p1 = Player(1, 'red', b)
    p2 = Player(2, 'yellow', b)
    g = Game(b, p1, p2)
    g.play_a_game()
