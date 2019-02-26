import random
import numpy as np
import torch
import torch.nn as nn
from src.networks import NeuralNet
from src.NNPlayer import NNPlayer
from src.NNGame import NNRecordedGame
from src.Board import Board


def sample_player_moves(nn_game, player, batch_size):
    input_data, output_data = [], []
    games_played = len(nn_game.history[player.name]['moves'])
    sampled_indices = random.sample(games_played, batch_size / 2)
    for j in sampled_indices:
        input_data.append((nn_game.history[player.name]['states'][j]))
        output_data.append(nn_game.history[player.name]['rollout_pol'][j])
    return input_data, output_data


def train(input_size: int, hidden_size: int, num_classes: int,
          num_epochs: int, batch_size: int, learning_rate: float,
          mcts_iter: int):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):

        # Prepare the game
        b = Board()

        # use output = model(stage) to get P
        p1 = NNPlayer(1, b, model, training=True)
        p2 = NNPlayer(2, b, model, training=True)
        nn_g = NNRecordedGame(b, p1, p2, mcts_iter, device)
        nn_g.initialize_history()

        # Play the game
        nn_g.play_a_game()

        # Sample batch_size moves
        input_p1, output_p1 = sample_player_moves(nn_g, p1, batch_size)
        input_p2, output_p2 = sample_player_moves(nn_g, p1, batch_size)

        # build the batch
        batch = [torch.tensor(input_p1 + input_p2),
                 torch.tensor(output_p1 + output_p2)]

        for i, (x, y) in enumerate(batch):
            # Move tensors to the configured device
            #x = x.reshape(-1, 6*7*2).to(device)
            x = x.to(device, dtype=torch.float64)
            y = y.to(device, dtype=torch.float64)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}] Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, loss.item()))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.reshape(-1, 28 * 28).to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
