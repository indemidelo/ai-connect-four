#from src.train import train
import tensorflow as tf
from src.Board import Board
from src.HumanPlayer import HumanPlayer
from src.tfPlayer import tfPlayer
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
    n_res_blocks = 1
    num_epochs = 2
    num_games = 1
    batch_size = 100
    learning_rate = 0.001
    mcts_iter = 1
    _, pred, inputs = train(
        n_res_blocks, num_epochs, num_games,
        batch_size, learning_rate, mcts_iter)
    # model.save('my_little_model.h5')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('\n\nSample Game\n\n')
        b = Board()
        p1 = tfPlayer(1, b, sess, pred, inputs, training=False)
        p2 = tfPlayer(2, b, sess, pred, inputs, training=False)

        nn_g = NNRecordedGame(b, p1, p2, 1)
        nn_g.initialize()
        nn_g.play_a_game(print_board=True)
