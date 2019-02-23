import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense
import keras
from src.NNPlayer import NNPlayer
from src.NNGame import NNRecordedGame
from src.Board import Board


def sample_player_moves(nn_game, player, batch_size):
    input_data, output_data = [], []
    games_played = len(nn_game.history[player.name]['moves'])
    sampled_indices = random.sample(range(games_played), int(batch_size / 2))
    for j in sampled_indices:
        input_data.append((nn_game.history[player.name]['states'][j]))
        output_data.append(nn_game.history[player.name]['rollout_pol'][j])
    input_data = np.asarray(input_data).reshape((4, 2, 6, 7))
    output_data = np.asarray(input_data).reshape((4, 2, 6, 7))
    return input_data, output_data


def train(input_size: int, hidden_size: int, num_classes: int,
          num_epochs: int, batch_size: int, learning_rate: float,
          mcts_iter: int):
    # Device configuration

    model = Sequential()
    #model.add(Conv2D(hidden_size, kernel_size=(3, 3), activation='relu',
                     #input_shape=(2, 6, 7), data_format='channels_first'))
    #model.add(BatchNormalization())
    #model.add(Flatten())
    model.add(Dense(14))
    model.add(Dense(7, kernel_initializer='normal'))

    model.compile(loss=keras.losses.mean_squared_error, optimizer='adam', metrics=['accuracy'])

    # Train the model
    for epoch in range(num_epochs):

        # Prepare the game
        b = Board()

        # use output = model(stage) to get P
        p1 = NNPlayer(1, b, model, training=True)
        p2 = NNPlayer(2, b, model, training=True)
        nn_g = NNRecordedGame(b, p1, p2, mcts_iter, device=None)
        nn_g.initialize_history()

        # Play the game
        nn_g.play_a_game()

        # Sample batch_size moves
        input_p1, output_p1 = sample_player_moves(nn_g, p1, 8)
        input_p2, output_p2 = sample_player_moves(nn_g, p2, 8)

        # build the batch
        batch = [np.concatenate([input_p1, input_p2]), np.concatenate([output_p1, output_p2])]

        model.fit(batch[0], batch[1], batch_size=batch_size, epochs=num_epochs, verbose=1)
