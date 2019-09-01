import os
from typing import List

import numpy as np
import tensorflow as tf

import settings
from game_engine import EMPTY
from player_ai import PlayerAIGameContext, PlayerAI, PlayerAIMove, MIN_QUALITY_VALUE, MAX_QUALITY_VALUE, log, \
    IMPOSSIBLE_PREDICTION, WIN


class DNNRegressorPlayerGameContext(PlayerAIGameContext):
    _player_ai: PlayerAI
    _player_ai_moves: List[PlayerAIMove]
    _base_learning_rate: float
    _earlier_move_learning_rate_delay: float
    _nudge_factor: float

    def __init__(self, player_ai: PlayerAI,
                 base_learning_rate: float,
                 earlier_move_learning_rate_delay: float,
                 nudge_factor: float = 0.05):
        self._nudge_factor = nudge_factor
        self._player_ai = player_ai
        self._base_learning_rate = base_learning_rate
        self._earlier_move_learning_rate_delay = earlier_move_learning_rate_delay
        self._player_ai_moves = []

    @property
    def player_name(self) -> str:
        return self._player_ai.player_name

    def get_next_move(self, standardised_game_state: np.array) -> int:
        """

        :param standardised_game_state: I.e. game state from this player's point of view
        :return: index of the next move
        """
        flat_game_state = standardised_game_state.flatten()

        original_prediction_array = self._player_ai.predict(flat_game_state=flat_game_state)

        prediction_array = original_prediction_array.copy()
        prediction_array = np.clip(prediction_array,
                                   a_min=MIN_QUALITY_VALUE,
                                   a_max=MAX_QUALITY_VALUE)

        while True:
            max_indices = np.where(prediction_array == np.amax(prediction_array))
            np.random.shuffle(max_indices)
            # Just take the first one after the shuffle
            try:
                next_move_index = max_indices[0][0]
            except IndexError:
                log(f'max_indices {max_indices}')
                raise
            # This should never be the maximum again
            prediction_array[next_move_index] = IMPOSSIBLE_PREDICTION
            # check valid move...
            if flat_game_state[next_move_index] == EMPTY:
                break
            else:
                log(f'Prediction for {self._player_ai.player_name} - Not a valid move, try again...')
                log(f'{max_indices}')
                log(f'{prediction_array}')

        self._player_ai_moves.append(
            PlayerAIMove(
                flat_game_state=flat_game_state,
                predictions=original_prediction_array,
                move_index=next_move_index
            )
        )

        log(f'{self.player_name} moving to {next_move_index}')

        # Get the highest value output node that is a valid mode
        return next_move_index

    def process_game_result(self, win_or_lose):
        """
        If win, weight winning moves positively
        If lose, weight losing moves negatively
        :param win_or_lose:
        :return:
        """
        # Positive learning rate if we win, negative if we lose
        nudge_factor = self._nudge_factor if win_or_lose == WIN else -self._nudge_factor
        # Start learning from the last move and work our way backwards...
        moves_copy = list(self._player_ai_moves)
        moves_copy.reverse()

        learning_rate = self._base_learning_rate

        for player_ai_move in moves_copy:
            # Adjust the target prediction by a nudge factor
            target_predictions = player_ai_move.predictions.copy()

            # Nudge our move quality value...
            target_predictions[player_ai_move.move_index] += nudge_factor
            target_predictions = np.clip(target_predictions,
                                         a_min=MIN_QUALITY_VALUE,
                                         a_max=MAX_QUALITY_VALUE)

            self._player_ai.train(
                flat_game_state=player_ai_move.flat_game_state,
                target_predictions=target_predictions,
                learning_rate=learning_rate
            )
            learning_rate *= self._earlier_move_learning_rate_delay


class DNNRegressorPlayer(PlayerAI):
    _base_learning_rate: float
    _player_name: str
    _earlier_move_learning_rate_delay: float

    def __init__(self, player_name: str,
                 base_learning_rate: float,
                 earlier_move_learning_rate_delay: float):
        self._player_name = player_name
        self._base_learning_rate = base_learning_rate
        self._earlier_move_learning_rate_delay = earlier_move_learning_rate_delay

        self._feature_columns = []

        self._num_cells = settings.NUM_ROWS * settings.NUM_COLS

        for index in range(0, self._num_cells):
            self._feature_columns.append(
                tf.feature_column.numeric_column(key=f'{index}')
            )

    def _create_dnn(self, learning_rate):
        # TODO work out a better way to adjust learning rate rather than recreate the DNN
        # every time we want to use it...
        return tf.estimator.DNNRegressor(hidden_units=settings.HIDDEN_UNITS,
                                         label_dimension=self._num_cells,
                                         model_dir=os.path.join(settings.MODEL_DIR, self._player_name),
                                         optimizer=tf.train.GradientDescentOptimizer(
                                             learning_rate=learning_rate
                                         ),
                                         feature_columns=self._feature_columns)

    @property
    def player_name(self):
        return self._player_name

    def predict(self, flat_game_state: np.array) -> np.array:
        """
        :param standardised_game_state: I.e. game state from this player's point of view
        :return: index of the next move
        """
        dnn = self._create_dnn(learning_rate=self._base_learning_rate)

        def input_fn():
            features = {}

            for index, element in enumerate(flat_game_state):
                key = f'{index}'
                prev_array = features.setdefault(key, np.array([]))
                features[key] = np.append(prev_array, [int(element)])

            return tf.data.Dataset.from_tensor_slices((features)).batch(1)

        prediction = list(dnn.predict(input_fn=input_fn))

        if len(prediction) != 1:
            raise ValueError('expected prediction to be length 1')

        log(f'Prediction array for {self.player_name}\n{prediction}')
        return prediction[0]['predictions']

    def train(self, flat_game_state: np.array, target_predictions: np.array, learning_rate=float):
        log(f'Training...\n{flat_game_state}\n{target_predictions}\n{learning_rate}\n')

        dnn = self._create_dnn(learning_rate=learning_rate)

        def train_fn():
            features = {}

            for index, element in enumerate(flat_game_state):
                key = f'{index}'
                prev_array = features.setdefault(key, np.array([]))
                features[key] = np.append(prev_array, [int(element)])

            labels = tf.constant(target_predictions, shape=[1, self._num_cells])

            return tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)

        dnn.train(input_fn=train_fn,
                  steps=1)

    def get_new_game_context(self) -> PlayerAIGameContext:
        return DNNRegressorPlayerGameContext(
            player_ai=self,
            base_learning_rate=self._base_learning_rate,
            earlier_move_learning_rate_delay=self._earlier_move_learning_rate_delay
        )
