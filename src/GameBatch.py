import numpy as np
from src.RecordedGame import RecordedGame
from src.Player import Player
from src.Board import Board


class GameBatch():
    def __init__(self, player_one, player_two, n_games):
        self.player_one = player_one
        self.player_two = player_two
        self.n_games = n_games
        self.states = np.empty((0, 6, 7))
        self.actions = np.empty((0,))
        self.results = {self.player_one.name: 0,
                        self.player_two.name: 0}

    def fire(self):
        for j in range(self.n_games):
            print(f'Playing Game {j+1}')
            b = Board()
            self.player_one.board = b
            self.player_two.board = b
            rg = RecordedGame(b, self.player_one, self.player_two)
            rg.initialize()
            rg.play_a_game()
            print(f'Player {rg.winner} won')
            if rg.winner is not None:
                self.results[rg.winner] += 1
            actions, states, res = rg.export_final_results()
            if res is 'ok' and b.plays > 3:
                dim_s = states.shape[0]
                last_states = states[np.array([dim_s-1])]
                self.states = np.append(self.states, last_states, axis=0)
                last_actions = actions[np.array([dim_s-1])]
                self.actions = np.append(self.actions, last_actions, axis=0)
                # self.states = np.append(self.states, states, axis=0)
                # self.actions = np.append(self.actions, actions, axis=0)
        print(f'Player {self.player_one.name} won {self.results[self.player_one.name]} time'
              f's\nPlayer {self.player_two.name} won {self.results[self.player_two.name]} tim'
              f'es')

    def export_final_results(self):
        np.save(f'../states_endgames.npy', np.array(self.states))
        np.save(f'../actions_endgames.npy', np.array(self.actions))


if __name__ == '__main__':
    b = Board()
    p1 = Player(1, b)
    p2 = Player(2, b)
    n_games = int(1e4)
    batch = GameBatch(p1, p2, n_games)
    batch.fire()
    batch.export_final_results()
