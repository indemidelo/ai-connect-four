#from src.train import train
from src.Board import Board
from src.HumanPlayer import HumanPlayer
from src.NNPlayer import NNPlayer
from src.NNGame import NNRecordedGame
from src.Game import Game
from src.RLPlayer import RLPlayer
from src.train_tensorflow import train

if __name__ == '__main__human':
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
    num_epochs = 25
    num_games = 250
    batch_size = 100
    learning_rate = 0.001
    mcts_iter = 250
    model = train(input_size, hidden_size, num_classes, num_epochs,
                  num_games, batch_size, learning_rate, mcts_iter)
    model.save('my_little_model.h5')

    print('\n\nSample Game\n\n')
    b = Board()
    p1 = NNPlayer(1, b, model, training=False)
    p2 = NNPlayer(2, b, model, training=False)

    nn_g = NNRecordedGame(b, p1, p2, mcts_iter)
    nn_g.initialize()
    nn_g.play_a_game(print_board=True)
