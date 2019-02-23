#from src.train import train
from src.Board import Board
from src.HumanPlayer import HumanPlayer
from src.Game import Game
from src.RLPlayer import RLPlayer
from src.train_tensorflow import train

if __name__ == '__main__Human':
    print('Human plays as 1 and PC as 2')
    b = Board()
    human = HumanPlayer(1, b)
    ai = RLPlayer(2, b, 250)
    g = Game(b, human, ai)
    g.play_a_game(print_board=True)

if __name__ == '__main__':
    print('Neural Network Training')
    # Hyper-parameters
    input_size = 2
    hidden_size = 256
    num_classes = 7
    num_epochs = 5
    batch_size = 4
    learning_rate = 0.001
    mcts_iter = 50
    train(input_size, hidden_size, num_classes, num_epochs,
          batch_size, learning_rate, mcts_iter)



