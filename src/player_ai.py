import abc
from typing import NamedTuple

import numpy as np

IMPOSSIBLE_PREDICTION = -999

MIN_QUALITY_VALUE = 0
MAX_QUALITY_VALUE = 1

WIN = 'win'
LOSS = 'loss'

TrainMove = NamedTuple('TrainMove',
                       game_state=np.array,
                       move=int,
                       adjustment=float
                       )

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
    def get_next_move(self, standardised_game_state: np.array) -> int:
        pass

    @abc.abstractmethod
    def process_game_result(self, win_or_lose):
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


