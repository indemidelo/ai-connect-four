from src.Board import Board
from src.HumanPlayer import HumanPlayer
from src.Game import Game
from src.RLPlayer import RLPlayer

if __name__ == '__main__':
    print('Human plays as 1 and PC as 2')
    b = Board()
    human = HumanPlayer(1, b)
    ai = RLPlayer(2, b, 250)
    g = Game(b, human, ai)
    g.play_a_game(print_board=True)
