import numpy as np
import tensorflow as tf

import settings
from game_engine import convert_move_to_index


def create_estimator():
    # OK, begin training
    feature_columns = []

    for index in range(0, settings.NUM_ROWS * settings.NUM_COLS):
        feature_columns.append(
            tf.feature_column.numeric_column(key=f'{index}')
        )

    return tf.estimator.DNNClassifier(hidden_units=settings.HIDDEN_UNITS,
                                      model_dir=settings.MODEL_DIR,
                                      feature_columns=feature_columns,
                                      n_classes=settings.NUM_COLS * settings.NUM_ROWS)


def get_train_fn(move_histories):
    def fn():
        labels = np.array([])
        features = {}

        for m_h in move_histories:
            for index, element in enumerate(m_h.state.flatten()):
                key = f'{index}'
                prev_array = features.setdefault(key, np.array([]))
                features[key] = np.append(prev_array, [int(element)])

            labels = np.append(labels, [convert_move_to_index(m_h.move, num_cols=settings.NUM_COLS)])

        dataset = tf.data.Dataset.from_tensor_slices((features, labels.astype(int)))
        dataset = dataset.shuffle(len(labels)).repeat().batch(len(labels))
        return dataset

    return fn


if __name__ == '__main__':
    from self_play import self_play, print_self_play_results

    estimator = create_estimator()

    self_play_result = self_play(estimator=estimator,
                                 num_games=1,
                                 num_cols=settings.NUM_COLS,
                                 num_rows=settings.NUM_ROWS,
                                 vs_random=False)

    print_self_play_results(self_play_result)

    estimator.train(input_fn=get_train_fn(
        move_histories=self_play_result.move_histories),
        steps=1
    )
