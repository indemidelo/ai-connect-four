from src.Move import Move


class HumanPlayer():
    def __init__(self, name, board):
        self.name = name
        self.board = board

    def play(self, fixed_move=None):
        col = self.human_move()
        if col != -1:
            m = Move(self.name, self.board, col)
            m.play()
            win = self.board.check_connect(m)
            return m, win
        else:
            self.board.playing = False
            self.board.full = True

    def search_available_moves(self):
        av_moves = list()
        for j in range(self.board.board.shape[1]):
            if 0 in self.board.board[:, j]:
                av_moves.append(j)
        return av_moves

    def human_move(self):
        available_moves = self.search_available_moves()
        if not available_moves:
            return -1
        try:
            col = int(input('Your move:')) - 1
        except:
            print('Invalid move! Column not found')
            return self.human_move()
        if col in available_moves:
            return col
        print('Invalid move! The column is full')
        return self.human_move()
