class Game():
    def __init__(self, board, player_one, player_two):
        """
        A game of n-connect
        :param board:
        :param player_one:
        :param player_two:
        """
        self.board = board
        self.player_one = player_one
        self.player_two = player_two
        self.winner = None
        self.board.playing = True

    def play_a_game(self, print_board=False, first_move=None):
        """ To play a game """
        while self.board.playing and not self.board.full:
            if print_board:
                print(self.board)
            if first_move is not None:
                p1move, win = self.player_one.play(first_move)
                first_move = None
            else:
                p1move, win = self.player_one.play()
            if win:
                self.winner = self.player_one.name
                self.board.playing = False
            elif not self.board.full:
                p2move, win = self.player_two.play()
                if win:
                    self.winner = self.player_two.name
                    self.board.playing = False
        if print_board:
            print(self.board)

    def one_move(self, move):
        if self.board.playing and not self.board.full:
            p1move, win = self.player_one.play(move)
            if win:
                self.winner = self.player_one.name
                self.board.playing = False
