from typing import List, NamedTuple

import numpy as np
import tensorflow as tf

from game_engine import MoveHistory, TicTacToeGameEngine, DRAW, TIC, TAC
from learn import create_estimator
from learn_utils import standardise_game_state, standardise_move_history
from settings import NUM_ROWS, NUM_COLS

SelfPlayResult = NamedTuple('SelfPlayResult',
                            tic_win=float,
                            tac_win=float,
                            draw=float,
                            illegal_predictions=int,
                            move_histories=List[MoveHistory])


def log(msg):
    print(msg)


def self_play(estimator: tf.estimator.DNNClassifier, num_games, num_cols, num_rows, vs_random) -> SelfPlayResult:
    move_histories: List[MoveHistory] = []
    tic_wins = 0
    tac_wins = 0
    draws = 0
    illegal_predictions = 0
    for i in range(0, num_games):
        log(f'Playing game {i}')

        engine = TicTacToeGameEngine(num_columns=num_cols, num_rows=num_rows)

        while not engine.is_game_over:

            def input_fn():
                features = {}

                for index, element in enumerate(
                        standardise_game_state(
                            state=engine.state,
                            my_move=engine.next_move).flatten()):
                    key = f'{index}'
                    prev_array = features.setdefault(key, np.array([]))
                    features[key] = np.append(prev_array, [int(element)])

                return tf.data.Dataset.from_tensor_slices((features)).batch(1)

            prediction = list(estimator.predict(input_fn=input_fn))

            if len(prediction) != 1:
                raise ValueError('expected prediction to be length 1')

            move_index = int(prediction[0]['class_ids'][0])
            if engine.is_valid_move_by_index(index=move_index):
                engine.do_next_move_by_flat_index(index=move_index)
            else:
                illegal_predictions += 1
                random_valid_move = engine.get_random_valid_move()
                engine.do_next_move(move=random_valid_move)

        if engine.game_result == DRAW:
            draws += 1
            # Get both game histories...
            move_histories += standardise_move_history(engine.get_move_history_for_player(TIC), TIC)
            move_histories += standardise_move_history(engine.get_move_history_for_player(TAC), TAC)
        else:
            winner = engine.game_result
            if winner == TIC:
                tic_wins += 1
            elif winner == TAC:
                tac_wins += 1
            else:
                raise ValueError(f'unexpected winner value {winner}')

            move_histories += standardise_move_history(engine.get_move_history_for_player(move=winner), winner)

    return SelfPlayResult(
        tic_win=tic_wins / float(num_games),
        tac_win=tac_wins / float(num_games),
        draw=draws / float(num_games),
        illegal_predictions=illegal_predictions,
        move_histories=move_histories
    )


def print_self_play_results(self_play_result: SelfPlayResult) -> None:
    print('tic win', self_play_result.tic_win)
    print('tac win', self_play_result.tac_win)
    print('draw', self_play_result.draw)
    print('illegal predictions', self_play_result.illegal_predictions)


if __name__ == '__main__':
    estimator = create_estimator()

    my_self_play_result = self_play(estimator=estimator,
                                    num_games=10,
                                    num_cols=NUM_COLS,
                                    num_rows=NUM_ROWS,
                                    vs_random=False)

    print_self_play_results(my_self_play_result)
