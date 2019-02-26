import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense
import keras
from src.NNPlayer import NNPlayer
from src.NNGame import NNRecordedGame
from src.Board import Board


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


def model_structure(hidden_size, input_shape):
    model = Sequential()
    model.add(Conv2D(hidden_size, kernel_size=(4, 4), activation='relu',
                     padding='same', input_shape=input_shape,
                     kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(hidden_size, kernel_size=(4, 4),
                     padding='same', activation='relu',
                     kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Conv2D(hidden_size, kernel_size=(4, 4),
                     padding='same', activation='relu',
                     kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(7, kernel_initializer='normal'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam', metrics=['accuracy'])
    return model


def train(input_size: int, hidden_size: int, num_classes: int,
          num_epochs: int, num_games: int, batch_size: int,
          learning_rate: float, mcts_iter: int):
    # Create the model
    input_shape = (6, 7, 2)
    model = model_structure(hidden_size, input_shape)

    # Train the model
    for e in range(num_games):

        # Create the board
        b = Board()

        # Create the players and the game
        p1 = NNPlayer(1, b, model, training=True)
        p2 = NNPlayer(2, b, model, training=True)
        nn_g = NNRecordedGame(b, p1, p2, mcts_iter, device=None)
        nn_g.initialize_history()

        # Play the game
        nn_g.play_a_game(True)

        # Sample batch_size moves
        if nn_g.winner == p1.name:
            input_data, output_data = sample_player_moves(nn_g, p1, 8)
        elif nn_g.winner == p2.name:
            input_data, output_data = sample_player_moves(nn_g, p1, 8)
        else:
            input_p1, output_p1 = sample_player_moves(nn_g, p1, 8)
            input_p2, output_p2 = sample_player_moves(nn_g, p2, 8)
            input_data = np.concatenate([input_p1, input_p2])
            output_data = np.concatenate([output_p1, output_p2])

        # build the batch
        batch = [input_data, output_data]

        # print('input data', input_data)
        # print('output data', output_data)
        model.fit(batch[0], batch[1], batch_size=batch_size,
                  epochs=num_epochs * (e + 1),
                  initial_epoch=num_epochs * e,
                  verbose=1)

    return model
