import sys

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context
import numpy as np

from VOClabelcolormap import color_map
from common import blend_from_3d_one_hots


class MyTensorBoardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, args, test_input_fn, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch',
                 profile_batch=2, embeddings_freq=0, embeddings_metadata=None, **kwargs):
        super().__init__(log_dir, histogram_freq, write_graph, write_images, update_freq, profile_batch,
                         embeddings_freq, embeddings_metadata, **kwargs)

        self.args = args

        self.cmap = color_map()
        self.last_training_epoch = 0
        self.last_test_image_prediction_epoch = -1

        it = iter(test_input_fn)

        self.images_to_predict = next(it).numpy()

    @staticmethod
    def _parse_args(args):
        header_row = 'Parameter | Value\n' \
                     '----------|------\n'

        args_dict = vars(args)

        # TODO: Move logic below out of this class
        if not args_dict['comment']:
            del args_dict['comment']

        if not args_dict['use_early_stopping']:
            del args_dict['early_stopping_patience']
            del args_dict['early_stopping_min_delta']

        if not args_dict['reduce_on_plateu_patience']:
            del args_dict['reduce_on_plateu_factor']
            del args_dict['reduce_on_plateu_epsilon']

        sys_arguments = ["{}=\"{}\"".format(argument.split('=')[0], argument.split('=')[1])
                         if " " in argument and "=" in argument
                         else argument
                         for argument in sys.argv[1:]]
        args_dict['command'] = 'python train.py {}'.format(" ".join(sys_arguments))

        table_body = ["{} | {}".format(key, value) for key, value in args_dict.items()]

        markdown = header_row + "\n".join(table_body)
        return markdown

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)

        self.last_training_epoch = epoch

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        writer_name = self._train_run_name
        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                writer = self._get_writer(writer_name)
                with writer.as_default():
                    tensor = tf.convert_to_tensor(self._parse_args(self.args))
                    tf.summary.text("run_settings", tensor, step=1)

    def on_test_batch_begin(self, batch, logs=None):
        super().on_test_batch_begin(batch, logs)

        if self.last_training_epoch == self.last_test_image_prediction_epoch:
            return

        annotations = self.model.predict(self.images_to_predict, steps=1)

        blended_examples = [blend_from_3d_one_hots(image, annotation, self.cmap, alpha=0.5)
                            for image, annotation
                            in zip(self.images_to_predict, annotations)]

        writer_name = self._validation_run_name
        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                writer = self._get_writer(writer_name)
                with writer.as_default():
                    tensor = tf.convert_to_tensor(np.array(blended_examples))
                    tf.summary.image("test_image", tensor, step=self.last_training_epoch, max_outputs=len(blended_examples))

        self.last_test_image_prediction_epoch = self.last_training_epoch
