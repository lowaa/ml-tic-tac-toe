from typing import List

import numpy as np

from game_engine import TIC, MoveHistory, TAC

# Standardising X's and O's for use in neural net
MINE = 4
THEIRS = 5


def standardise_game_state(state: np.array, my_move: int):
    result = state.copy()

    if my_move == TIC:
        their_move = TAC
    elif my_move == TAC:
        their_move = TIC
    else:
        raise ValueError('my_move is invalid')

    result[result == my_move] = MINE
    result[result == their_move] = THEIRS

    return result


def standardise_move_history(move_history: List[MoveHistory], my_move: int) -> [MoveHistory]:
    """
    This returns the board state and the move that the player took, for every stage of the game.
    This will always make empty squares, EMPTY
    The player's squares will be 0
    The opponent's square will be 1
    :param move_history:
    :param move:
    :return:
    """
    result = []
    for m_h in move_history:
        result.append(MoveHistory(
            state=standardise_game_state(state=m_h.state, my_move=my_move),
            move=m_h.move
        ))

    return result
