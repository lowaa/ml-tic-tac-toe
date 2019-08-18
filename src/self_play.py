from typing import List, NamedTuple

from game_engine import MoveHistory, TicTacToeGameEngine, DRAW, TIC, TAC
from learn import create_estimator
from learn_utils import standardise_game_state, standardise_move_history
from player_ai import PlayerAI, DNNRegressorPlayer, WIN, LOSS
from settings import NUM_ROWS, NUM_COLS

SelfPlayResult = NamedTuple('SelfPlayResult',
                            tic_win=float,
                            tac_win=float,
                            draw=float,
                            illegal_predictions=int,
                            move_histories=List[MoveHistory])


def log(msg):
    print(msg)


def self_play(player_tic: PlayerAI, player_tac: PlayerAI,
              num_games: int, num_cols: int, num_rows: int) -> SelfPlayResult:
    move_histories: List[MoveHistory] = []
    tic_wins = 0
    tac_wins = 0
    draws = 0
    illegal_predictions = 0

    for i in range(0, num_games):
        log(f'Playing game {i}')

        engine = TicTacToeGameEngine(num_columns=num_cols, num_rows=num_rows)

        player_tic_game_context = player_tic.get_new_game_context()
        player_tac_game_context = player_tac.get_new_game_context()

        next_player_game_context = player_tic_game_context

        while not engine.is_game_over:

            standardised_game_state = standardise_game_state(
                state=engine.state,
                my_move=engine.next_move
            )

            move_index = next_player_game_context.get_next_move(standardised_game_state=standardised_game_state)

            if engine.is_valid_move_by_index(index=move_index):
                engine.do_next_move_by_flat_index(index=move_index)
            else:
                illegal_predictions += 1
                random_valid_move = engine.get_random_valid_move()
                engine.do_next_move(move=random_valid_move)

            next_player_game_context = player_tic_game_context if engine.next_move == TIC else player_tac_game_context

        if engine.game_result == DRAW:
            draws += 1

            # Lets just count a draw as win
            player_tic_game_context.process_game_result(win_or_lose=WIN)
            # player_tac_game_context.process_game_result(win_or_lose=WIN)

            # Get both game histories...
            move_histories += standardise_move_history(engine.get_move_history_for_player(TIC), TIC)
            move_histories += standardise_move_history(engine.get_move_history_for_player(TAC), TAC)
        else:
            winner = engine.game_result
            if winner == TIC:
                tic_wins += 1
                player_tic_game_context.process_game_result(win_or_lose=WIN)
                # player_tac_game_context.process_game_result(win_or_lose=LOSS)

            elif winner == TAC:
                tac_wins += 1
                player_tic_game_context.process_game_result(win_or_lose=LOSS)
                # player_tac_game_context.process_game_result(win_or_lose=WIN)
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

    player_tic = DNNRegressorPlayer(player_name='bob',
                                    base_learning_rate=0.4,
                                    earlier_move_learning_rate_delay=0.5)
    player_tac = DNNRegressorPlayer(player_name='carol',
                                    base_learning_rate=0.4,
                                    earlier_move_learning_rate_delay=0.5)

    my_self_play_result = self_play(player_tic=player_tic,
                                    player_tac=player_tac,
                                    num_games=50,
                                    num_cols=NUM_COLS,
                                    num_rows=NUM_ROWS)

    print_self_play_results(my_self_play_result)
