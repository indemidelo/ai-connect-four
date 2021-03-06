#import torch
import numpy as np


class bcolors:
    CYANO = '\033[96m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


class Board():
    def __init__(self, rows=6, columns=7):
        self.board = np.zeros((rows, columns), dtype=float)
        self.playing = False
        self.plays = 0
        self.full = False

    def __repr__(self):
        print()
        print('-------------------------------')
        print(f'-----------{bcolors.HEADER}THE BOARD{bcolors.ENDC}-----------')
        print('-------------------------------')
        for j in self.board:
            print(' |', end=' ')
            for i in j:
                value = f'{bcolors.OKBLUE}O {bcolors.ENDC}' if i == 1 else f'{bcolors.RED}X {bcolors.ENDC}' \
                    if i == 2 else f'{bcolors.GRAY}_ {bcolors.ENDC}'
                # value = f'{i} ' if i != 0 else '_ '
                print(f'{value}|', end=' ')
            print()
        print('---1---2---3---4---5---6---7---')
        return ''

    def play_(self, player, col):
        if self.board[:, col][0] != 0:
            print(f'The {col + 1} column is already full')
            return -1
        else:
            pos = self.find_free_spot(col)
            self.board[pos, col] = float(player)
            self.plays += 1
            if self.plays == self.board.shape[0] * self.board.shape[1]:
                self.full = True
                self.playing = False
            return pos

    def find_free_spot(self, col):
        column = self.board[:, col]
        return max([j for j, v in enumerate(column) if v == 0])

    # def color_encoder(self, player):
    #     if player == 'O':
    #         return 1
    #     elif player == 'X':
    #         return 2

    def check_connect(self, last_move):
        if self.horizontal_connect(last_move):
            # return f'Player {last_move.player} wins'
            return True
        if self.vertical_connect(last_move):
            # return f'Player {last_move.player} wins'
            return True
        if self.diagonal_connect(last_move):
            # return f'Player {last_move.player} wins'
            return True
        return False

    def winning_combo(self, combo, player):
        # print('combo', combo, 'player', player)
        return all(j == player for j in combo)

    def horizontal_connect(self, move):
        combo_start = max(0, move.col - 3)
        combo_end = min(6, move.col + 3)
        # print('Horizontal Combos')
        for j in range(combo_end - combo_start - 2):
            combo = self.board[move.row, combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, move.player):
                self.playing = False
                return True
        return False

    def vertical_connect(self, move):
        combo_start = max(0, move.row - 3)
        combo_end = min(5, move.row + 3)
        # print('Vertical Combos')
        for j in range(combo_end - combo_start - 2):
            combo = self.board[combo_start + j: combo_start + j + 4, move.col]
            if self.winning_combo(combo, move.player):
                self.playing = False
                return True
        return False

    def diagonal_connect(self, move):
        if self.diagonal_connect_nw_to_se(move):
            return True
        elif self.diagonal_connect_ne_to_sw(move):
            return True
        return False

    def diagonal_connect_nw_to_se(self, move):
        """
        North west to south east
        :param move:
        :return:
        """
        # combo_matrix = np.array([
        #     [0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 1, 1],
        #     [0, 1, 2, 2, 2, 2, 2],
        #     [0, 1, 2, 3, 3, 3, 3],
        #     [0, 1, 2, 3, 4, 4, 4],
        #     [0, 1, 2, 3, 4, 5, 5]])
        diagonal = self.board.diagonal(move.col - move.row)
        j_diag = min(move.col, move.row)
        combo_start = max(0, j_diag - 3)
        combo_end = min(len(diagonal) - 1, j_diag + 3)
        # print('Diagonal NW combos')
        for j in range(combo_end - combo_start - 2):
            combo = diagonal[combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, move.player):
                self.playing = False
                return True
        return False

    def diagonal_connect_ne_to_sw(self, move):
        """
        South west to north east
        :param move:
        :return:
        """
        specular_col = abs(6 - move.col)
        diagonal = np.flip(self.board, axis=-1).diagonal(specular_col - move.row)
        j_diag = min(specular_col, move.row)
        combo_start = max(0, j_diag - 3)
        combo_end = min(len(diagonal) - 1, j_diag + 3)
        # print('Diagonal NE combos')
        for j in range(combo_end - combo_start - 2):
            combo = diagonal[combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, move.player):
                self.playing = False
                return True
        return False

    def uniform_board(self):
        new_board = []
        for j in self.board:
            row = []
            for i in j:
                row.append(int(1 / (0.5 * i) if i else i))
            new_board.append(row)
        return new_board

    def list_available_moves(self) -> list:
        av_moves = list()
        for j in range(self.board.shape[1]):
            if 0 in self.board[:, j]:
                av_moves.append(j)
        return av_moves

    def input_data_board(self):
        p1board = []
        for j in self.board:
            row = []
            for i in j:
                row.append(1.0 if i == 1 else 0.0)
            p1board.append(row)
        p2board = []
        for j in self.board:
            row = []
            for i in j:
                row.append(1.0 if i == 2 else 0.0)
            p2board.append(row)
        return p1board, p2board

    def board_as_tensor(self, player):
        p1board, p2board = self.input_data_board()
        if player == 1:
            player_matrix = np.ones((6, 7))
            #return torch.tensor((p1board, player_matrix)).reshape(-1, 2, 6, 7)
            #return torch.tensor((p1board, player_matrix)).reshape(1, 2, 6, 7)
            return np.array((p1board, player_matrix)).reshape((1, 6, 7, 2))
        else:
            player_matrix = np.zeros((6, 7))
            #return torch.tensor((p2board, player_matrix)).reshape(1, 2, 6, 7)
            return np.array((p2board, player_matrix)).reshape((1, 6, 7, 2))



