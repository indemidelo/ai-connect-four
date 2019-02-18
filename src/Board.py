import numpy as np


class Board():
    def __init__(self):
        self.board = np.zeros((6, 7))
        self.playing = False

    def __repr__(self):
        print()
        print('-------------------------------')
        print('-----------THE BOARD-----------')
        print('-------------------------------')
        for j in self.board:
            print(' |', end=' ')
            for i in j:
                value = '1 ' if i == 1 else '2 ' if i == 2 else '_ '
                print(f'{value}|', end=' ')
            print()
        print('-------------------------------')
        return ''

    def play_(self, player, col):
        if self.board[:, col][0] != 0:
            print(f'The {col + 1} column is already full')
        else:
            pos = self.find_free_spot(col)
            self.board[pos, col] = player
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
            return f'{last_move.player} wins'
        if self.vertical_connect(last_move):
            return f'{last_move.player} wins'
        if self.diagonal_connect(last_move):
            return f'{last_move.player} wins'
        return None

    def winning_combo(self, combo, player):
        print('combo', combo, 'player', player)
        return all(j == player for j in combo)

    def horizontal_connect(self, move):
        combo_start = max(0, move.col - 3)
        combo_end = min(6, move.col + 3)
        print('Horizontal Combos')
        for j in range(combo_end - combo_start - 2):
            combo = self.board[move.row, combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, move.player):
                self.playing = False
                return True
        return False

    def vertical_connect(self, move):
        combo_start = max(0, move.row - 3)
        combo_end = min(5, move.row + 3)
        print('Vertical Combos')
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
        print('Diagonal NW combos')
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
        print('Diagonal NE combos')
        for j in range(combo_end - combo_start - 2):
            combo = diagonal[combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, move.player):
                self.playing = False
                return True
        return False
