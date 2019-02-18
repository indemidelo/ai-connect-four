from src.Board import Board
from src.Player import Player


class Game():
    def __init__(self, board, player_one, player_two):
        self.board = board
        self.player_one = player_one
        self.player_two = player_two

    def play_a_game(self):
        self.board.playing = True
        while self.board.playing:
            self.player_one.play()
            if self.board.playing:
                self.player_two.play()
            print(self.board)


if __name__ == '__main__':
    b = Board()
    p1 = Player(1, 'red', b)
    p2 = Player(2, 'yellow', b)
    g = Game(b, p1, p2)
    g.play_a_game()
