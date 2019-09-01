from typing import NamedTuple, List

import numpy as np
from matplotlib import pyplot as plt

from game_engine import TicTacToeGameEngine, DRAW, TIC, TAC
from learn import create_estimator
from learn_utils import standardise_game_state
from player_ai import PlayerAI, PLAYER_WIN, PLAYER_LOSS, PLAYER_DRAW
from player_ai_random import RandomPlayer
from player_ai_dnn_regressor import DNNRegressorPlayer
from settings import NUM_ROWS, NUM_COLS

SelfPlayResult = NamedTuple('SelfPlayResult',
                            player_1_win_percentage_series=List[float],
                            player_2_win_percentage_series=List[float],
                            draw_percentage_series=List[float],
                            player_1_wins=float,
                            player_2_wins=float,
                            draw=float,
                            illegal_predictions=int)


def log(msg):
    print(msg)


def self_play(player_1: PlayerAI,
              player_2: PlayerAI,
              num_games: int, num_cols: int, num_rows: int) -> SelfPlayResult:
    player_1_wins = 0
    player_2_wins = 0
    draws = 0
    illegal_predictions = 0
    player_1_win_percentage_series = []
    player_2_win_percentage_series = []
    draw_percentage_series = []

    for i in range(0, num_games):
        log(f'Playing game {i}')

        engine = TicTacToeGameEngine(num_columns=num_cols, num_rows=num_rows)

        player_1_game_context = player_1.get_new_game_context()
        player_2_game_context = player_2.get_new_game_context()

        coin_flip = np.random.randint(0, 2)

        if coin_flip == 0:
            player_tic_game_context = player_1_game_context
            player_tac_game_context = player_2_game_context
        else:
            player_tic_game_context = player_2_game_context
            player_tac_game_context = player_1_game_context

        next_player_game_context = player_1_game_context if coin_flip == 0 else player_2_game_context

        log(f'Coin flip is {coin_flip}. {next_player_game_context.player_name} goes first with {engine.next_move}')

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

            next_player_game_context = player_tac_game_context if next_player_game_context == player_tic_game_context \
                else player_tic_game_context

        if engine.game_result == DRAW:
            log('################### The game is a draw ####################')
            draws += 1

            # Lets just count a draw as win
            player_1_game_context.process_game_result(win_or_lose_or_draw=PLAYER_DRAW)
            player_2_game_context.process_game_result(win_or_lose_or_draw=PLAYER_DRAW)

        else:
            winner = engine.game_result
            if winner == TIC:
                log(f'################### {player_tic_game_context.player_name} wins! ####################')

                if player_tic_game_context == player_1_game_context:
                    player_1_wins += 1
                elif player_tic_game_context == player_2_game_context:
                    player_2_wins += 1
                else:
                    raise ValueError('unexpected player game context value')

                player_tic_game_context.process_game_result(win_or_lose_or_draw=PLAYER_WIN)
                player_tac_game_context.process_game_result(win_or_lose_or_draw=PLAYER_LOSS)

            elif winner == TAC:
                log(f'################### {player_tac_game_context.player_name} wins! ####################')

                if player_tac_game_context == player_1_game_context:
                    player_1_wins += 1
                elif player_tac_game_context == player_2_game_context:
                    player_2_wins += 1
                else:
                    raise ValueError('unexpected player game context value')

                player_tic_game_context.process_game_result(win_or_lose_or_draw=PLAYER_LOSS)
                player_tac_game_context.process_game_result(win_or_lose_or_draw=PLAYER_WIN)
            else:
                raise ValueError(f'unexpected winner value {winner}')

        # Update series data
        total_games = player_1_wins + player_2_wins + draws
        player_1_win_percentage_series.append(player_1_wins / float(total_games))
        player_2_win_percentage_series.append(player_2_wins / float(total_games))
        draw_percentage_series.append(draws / float(total_games))

    return SelfPlayResult(
        player_1_win_percentage_series=player_1_win_percentage_series,
        player_2_win_percentage_series=player_2_win_percentage_series,
        draw_percentage_series=draw_percentage_series,
        player_1_wins=player_1_wins / float(num_games),
        player_2_wins=player_2_wins / float(num_games),
        draw=draws / float(num_games),
        illegal_predictions=illegal_predictions
    )


def print_self_play_results(self_play_result: SelfPlayResult) -> None:
    print('player 1 win', self_play_result.player_1_wins)
    print('player 2 win', self_play_result.player_2_wins)
    print('draw', self_play_result.draw)
    print('illegal predictions', self_play_result.illegal_predictions)


def draw_self_play_results_plot(self_play_result: SelfPlayResult) -> None:
    total_games = len(self_play_result.player_1_win_percentage_series)
    game_number = list(range(1, int(total_games) + 1))

    plt.plot(game_number, self_play_result.player_1_win_percentage_series, 'r')
    plt.plot(game_number, self_play_result.player_2_win_percentage_series, 'b')
    plt.plot(game_number, self_play_result.draw_percentage_series, 'g')
    plt.xlabel('game number')
    plt.ylabel('percentage')
    plt.title('Self play result')
    plt.show()


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('num_games', type=int)

    args = arg_parser.parse_args()

    estimator = create_estimator()

    player_tic = DNNRegressorPlayer(player_name='bob',
                                    base_learning_rate=0.2,
                                    earlier_move_learning_rate_decay=0.95)

    # player_tac = DNNRegressorPlayer(player_name='carol',
    #                                 base_learning_rate=0.4,
    #                                 earlier_move_learning_rate_decay=0.5)

    player_tac = RandomPlayer(player_name='randy')

    my_self_play_result = self_play(player_1=player_tic,
                                    player_2=player_tac,
                                    num_games=args.num_games,
                                    num_cols=NUM_COLS,
                                    num_rows=NUM_ROWS)

    print_self_play_results(my_self_play_result)
    draw_self_play_results_plot(my_self_play_result)
