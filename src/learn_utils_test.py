import numpy as np
from hamcrest import assert_that, equal_to
from game_engine import MoveHistory, Move, TIC, TAC, EMPTY
from learn_utils import standardise_move_history, MINE, THEIRS


def test_standardise_move_history():
    move_history_list = [
        MoveHistory(move=Move(1, 1), state=np.array([[TIC, TAC, EMPTY]]))
    ]

    result = standardise_move_history(move_history=move_history_list,
                                      my_move=TIC)

    assert_that(result[0].state[0][0], equal_to(MINE))
    assert_that(result[0].state[0][1], equal_to(THEIRS))
    assert_that(result[0].state[0][2], equal_to(EMPTY))
