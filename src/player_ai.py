import abc
from typing import NamedTuple

import numpy as np


PLAYER_WIN = 'win'
PLAYER_LOSS = 'loss'
PLAYER_DRAW = 'draw'

PlayerAIMove = NamedTuple('PlayerMove',
                          flat_game_state=np.array,
                          predictions=np.array,
                          move_index=int)


def log(msg):
    print(msg)


class PlayerAIGameContext(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def player_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_next_move(self, standardised_game_state: np.array,
                      session_complete_percentage: float) -> int:
        """

        :param standardised_game_state:
        :param session_complete_percentage: number between 0 and 1
        :return:
        """
        pass

    @abc.abstractmethod
    def process_game_result(self, win_or_lose_or_draw: str):
        pass


class PlayerAI(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def player_name(self) -> str:
        pass

    @abc.abstractmethod
    def train(self, flat_game_state: np.array, target_predictions: np.array, learning_rate=float):
        pass

    @abc.abstractmethod
    def predict(self, flat_game_state: np.array) -> np.array:
        pass

    @abc.abstractmethod
    def get_new_game_context(self) -> PlayerAIGameContext:
        pass
