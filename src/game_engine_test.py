import unittest

from hamcrest import assert_that, equal_to, raises, calling

from game_engine import TicTacToeGameEngine, TIC, Move, convert_move_to_index


class TicTacToeGameEngineTestCase(unittest.TestCase):

    def test_column_win(self):
        subject = TicTacToeGameEngine(num_rows=3, num_columns=3, starting_move=TIC)

        subject.do_next_move(Move(row=0, col=0))  # TIC
        subject.do_next_move(Move(row=0, col=1))  # TOE
        subject.do_next_move(Move(row=1, col=0))  # TIC
        subject.do_next_move(Move(row=0, col=2))  # TOE

        assert_that(subject.is_game_over, equal_to(False))
        assert_that(subject.game_result, equal_to(None))

        # The winning move
        subject.do_next_move(Move(row=2, col=0))  # TIC

        assert_that(subject.is_game_over, equal_to(True))
        assert_that(subject.game_result, equal_to(TIC))

        # Once the game is over, no new moves are allowed
        assert_that(calling(subject.do_next_move).with_args(Move(2, 2)), raises(ValueError))

    def test_row_win(self):
        subject = TicTacToeGameEngine(num_rows=3, num_columns=3, starting_move=TIC)

        subject.do_next_move(Move(row=0, col=0))  # TIC
        subject.do_next_move(Move(row=1, col=0))  # TOE
        subject.do_next_move(Move(row=0, col=1))  # TIC
        subject.do_next_move(Move(row=1, col=1))  # TOE

        assert_that(subject.is_game_over, equal_to(False))
        assert_that(subject.game_result, equal_to(None))

        # The winning move
        subject.do_next_move(Move(row=0, col=2))  # TIC

        assert_that(subject.is_game_over, equal_to(True))
        assert_that(subject.game_result, equal_to(TIC))

    def test_top_left_bottom_right_diagonal_win(self):
        subject = TicTacToeGameEngine(num_rows=3, num_columns=3, starting_move=TIC)

        subject.do_next_move(Move(row=0, col=0))  # TIC
        subject.do_next_move(Move(row=1, col=0))  # TOE
        subject.do_next_move(Move(row=1, col=1))  # TIC
        subject.do_next_move(Move(row=1, col=2))  # TOE

        assert_that(subject.is_game_over, equal_to(False))
        assert_that(subject.game_result, equal_to(None))

        # The winning move
        subject.do_next_move(Move(row=2, col=2))  # TIC

        assert_that(subject.is_game_over, equal_to(True))
        assert_that(subject.game_result, equal_to(TIC))

    def test_illegal_move_error(self):
        subject = TicTacToeGameEngine(num_rows=3, num_columns=3, starting_move=TIC)

        subject.do_next_move(Move(row=0, col=0))  # TIC
        # Illegal move
        assert_that(calling(subject.do_next_move).with_args(Move(row=0, col=0)), raises(ValueError))

    def test_get_valid_moves(self):
        subject = TicTacToeGameEngine(num_rows=3, num_columns=3, starting_move=TIC)

        subject.do_next_move(Move(row=0, col=0))  # TIC
        subject.do_next_move(Move(row=0, col=1))  # TOE
        subject.do_next_move(Move(row=1, col=1))
        subject.do_next_move(Move(row=2, col=2))

        assert_that(subject.is_game_over, equal_to(False))

        valid_moves = subject.get_valid_moves()
        assert_that(valid_moves, equal_to([
            Move(row=0, col=2),
            Move(row=1, col=0),
            Move(row=1, col=2),
            Move(row=2, col=0),
            Move(row=2, col=1),
        ]))

        assert_that(subject.is_valid_move(Move(row=0, col=0)), equal_to(False))
        assert_that(subject.is_valid_move(Move(row=1, col=2)), equal_to(True))

        random_valid_move = subject.get_random_valid_move()
        assert_that(random_valid_move in valid_moves, equal_to(True))

    def test_convert_move_to_index(self):
        assert_that(convert_move_to_index(move=Move(0, 1), num_cols=3), equal_to(1))
        assert_that(convert_move_to_index(move=Move(2, 1), num_cols=3), equal_to(7))
