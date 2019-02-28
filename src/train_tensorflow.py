import random
import time
import numpy as np
import tensorflow as tf
from src.tfPlayer import tfPlayer
from src.NNGame import NNRecordedGame
from src.NNGame_mp import NNRecordedGame_mp
from src.Board import Board
from src.tensorflow_network import AlphaGo19Net


def sample_player_moves(nn_game, player, batch_size):
    input_data, output_data = [], []
    games_played = len(nn_game.history[player.name]['moves'])
    games_to_sample = min(games_played, int(batch_size / 2))
    sampled_indices = random.sample(range(games_to_sample), int(batch_size / 2))
    for j in sampled_indices:
        input_data.append((nn_game.history[player.name]['states'][j]))
        output_data.append(nn_game.history[player.name]['rollout_pol'][j])
    input_data = np.asarray(input_data).reshape((4, 6, 7, 2))
    output_data = np.asarray(output_data).reshape((4, 7))
    return input_data, output_data


def get_all_player_moves(nn_game, player):
    input_data = np.asarray(nn_game.history[player.name]['states'])
    input_data = input_data.reshape((-1, 6, 7, 2))
    output_data = np.asarray(nn_game.history[player.name]['rollout_pol'])
    output_data = output_data.reshape((-1, 7))
    return input_data, output_data


def train(n_res_blocks: int, num_epochs: int, num_games: int,
          batch_size: int, learning_rate: float, mcts_iter: int):

    # log directory
    logs_path = '/tmp/tensorflow_logs/example/'

    # Placeholder for input_data
    inputs = tf.placeholder(tf.float32, [None, 6, 7, 2], name='InputData')

    # Placeholder for p
    p = tf.placeholder(tf.float32, [None, 7], name='p')

    # Neural Network
    pred, loss, optimizer, acc = AlphaGo19Net(
        inputs, p, n_res_blocks, learning_rate)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    print("Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs \n"
          "Then open http://0.0.0.0:6006/ into your web browser")

    # Train the model
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for e in range(num_games):

            # Create the board
            b = Board()

            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

            # Create the players and the game
            p1 = tfPlayer(1, b, sess, pred, inputs, training=True)
            p2 = tfPlayer(2, b, sess, pred, inputs, training=True)
            #nn_g = NNRecordedGame_mp(b, p1, p2, mcts_iter)
            nn_g = NNRecordedGame(b, p1, p2, mcts_iter)
            nn_g.initialize()

            # Play the game
            nn_g.play_a_game(True)

            # Collect the results
            input_p1, output_p1 = get_all_player_moves(nn_g, p1)
            input_p2, output_p2 = get_all_player_moves(nn_g, p2)
            input_data = np.concatenate([input_p1, input_p2])
            output_data = np.concatenate([output_p1, output_p2])

            # Training cycle
            for epoch in range(num_epochs):
                # fit the model
                _, c, summary = sess.run(
                    [optimizer, loss, merged_summary_op],
                    feed_dict={inputs: input_data, p: output_data})

                # Write logs at every iteration
                summary_writer.add_summary(summary, e)

                if (epoch + 1) % 25 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            # save the model every 100 games played
            #if e and e % 100 == 0:
            #    model.save(f'{round(time.time())}_my_model.h5')
