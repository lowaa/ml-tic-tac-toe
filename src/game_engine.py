import random
from typing import NamedTuple

import numpy as np

import settings

EMPTY = 0
TIC = 7
TAC = 8

DRAW = 100

Move = NamedTuple('TicTacToeMove', row=int, col=int)

MoveHistory = NamedTuple('MoveHistory', state=np.array, move=Move)


class TicTacToeGameEngine:
    _num_rows: int
    _num_columns: int
    _first_move: int
    _next_move: int
    _state: np.array
    _state_history: [np.array]
    _game_result: int

    def __init__(self, num_rows=3, num_columns=3, starting_move=None):
        self._num_rows = num_rows
        self._num_columns = num_columns

        if starting_move is None:
            starting_move = np.random.randint(TIC, TAC, 1)[0]

        self._first_move = starting_move
        self._next_move = starting_move

        self._state = np.full(shape=(num_rows, num_columns), fill_value=EMPTY)

        # First state history is the empty board
        self._move_history = {
            TIC: [],
            TAC: []
        }

        self._game_result = None

    @property
    def state(self):
        return self._state

    def do_next_move_by_flat_index(self, index: int) -> int:
        return self.do_next_move(move=self._index_convert_to_move(index=index))

    def do_next_move(self, move: Move) -> int:
        """
        :return: The game result, if the next move resulted in game over
        """
        row = move.row
        col = move.col

        if self._game_result is None:
            if self._state[row][col] != EMPTY:
                raise ValueError('Illegal move')

            self._move_history[self._next_move].append(MoveHistory(
                state=self._state.copy(),
                move=move
            ))

            self._state[row][col] = self._next_move

            if settings.DRAW_BOARD:
                print("""
 {} | {} | {}
---+---+---
 {} | {} | {}
---+---+---
 {} | {} | {}
                """.format(*[' ' if x == EMPTY else x for x in self._state.flatten()]))

            # Check winner
            self._game_result = check_winner(state=self._state,
                                             last_move=self._next_move,
                                             num_rows=self._num_rows,
                                             num_cols=self._num_columns)

            self._next_move = self._get_the_other_move(self._next_move)

            return self._game_result
        else:
            raise ValueError(f'game is over. result is : {self._game_result}')

    @property
    def is_game_over(self) -> bool:
        return self._game_result is not None

    @property
    def game_result(self) -> int:
        return self._game_result

    @property
    def first_move(self) -> int:
        return self._first_move

    @property
    def next_move(self) -> int:
        return self._next_move

    def is_valid_move(self, move: Move) -> bool:
        return move in self.get_valid_moves()

    def is_valid_move_by_index(self, index: int) -> bool:
        move = self._index_convert_to_move(index=index)
        return self.is_valid_move(move=move)

    def get_valid_moves(self) -> [Move]:
        result = []
        for index, v in enumerate(self._state.flatten()):
            if v == EMPTY:
                result.append(self._index_convert_to_move(index))
        return result

    def get_random_valid_move(self):
        return random.choice(self.get_valid_moves())

    def get_move_history_for_player(self, move):
        """
        This returns the board state and the move that the player took, for every stage of the game
        :param move:
        :return:
        """
        if move not in self._move_history:
            raise ValueError(f'move {move} not in move history')
        return self._move_history[move]

    def _index_convert_to_move(self, index) -> Move:
        row = int(index / float(self._num_columns))
        col = index - row * self._num_columns
        return Move(row=row, col=col)

    def _get_the_other_move(self, move: int) -> int:
        """
        Just a helper to get the other move
        :param move:
        :return:
        """
        return TAC if move == TIC else TIC


def convert_move_to_index(move: Move, num_cols: int) -> int:
    return move.row * num_cols + move.col


def check_winner(state: np.array, last_move: int, num_rows: int, num_cols: int):
    # Borrowed from https://github.com/geoffreyyip/numpy-tictactoe
    # TODO: make this more generic so we can play more varied game types

    if num_rows != num_cols:
        raise ValueError('TODO: handle arbitrary board sizes')

    for i in range(0, 3):
        # Checks rows and columns for match
        rows_win = (state[i, :] == last_move).all()
        cols_win = (state[:, i] == last_move).all()

        if rows_win or cols_win:
            return last_move

    diag1_win = (np.diag(state) == last_move).all()
    diag2_win = (np.diag(np.fliplr(state)) == last_move).all()

    if diag1_win or diag2_win:
        # Checks both diagonals for match
        return last_move

    # Check for draw
    if not (state.flatten() == EMPTY).any():
        # We have a draw
        return DRAW
