import numpy as np

from game_engine import EMPTY
from player_ai import PlayerAIGameContext, PlayerAI, log


class RandomPlayerGameContext(PlayerAIGameContext):

    def __init__(self, player_ai: PlayerAI):
        self._player_ai = player_ai

    @property
    def player_name(self) -> str:
        return self._player_ai.player_name

    def get_next_move(self, standardised_game_state: np.array) -> int:
        flat_game_state = standardised_game_state.flatten()
        while True:
            next_move_index = np.random.randint(0, len(flat_game_state))
            # check valid move...
            if flat_game_state[next_move_index] == EMPTY:
                break
            else:
                log(f'Prediction for {self._player_ai.player_name} - Not a valid move, try again...')

        log(f'{self.player_name} moving to {next_move_index}')

        return next_move_index

    def process_game_result(self, win_or_lose_or_draw: str):
        pass


class RandomPlayer(PlayerAI):

    def __init__(self, player_name):
        self._player_name = player_name

    def train(self, flat_game_state: np.array, target_predictions: np.array, learning_rate=float):
        pass

    def predict(self, flat_game_state: np.array) -> np.array:
        pass

    def get_new_game_context(self) -> PlayerAIGameContext:
        return RandomPlayerGameContext(
            player_ai=self
        )

    @property
    def player_name(self) -> str:
        return self._player_name