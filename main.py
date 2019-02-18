from src.Board import Board
from src.Player import Player

if __name__ == '__main__':
    b = Board()
    p1 = Player(1, 'red', b)
    p2 = Player(2, 'yellow', b)
    p1.play(4)
    p2.play(2)
    print(b)
    p1.play(4)
    p2.play(3)
    print(b)
    #p1.play(3)
    #p2.play(3)
    #print(b)
