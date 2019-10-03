import argparse
import os
from datetime import datetime
from glob import glob

import tensorflow as tf

import common
from callbacks.TensorBoard import MyTensorBoardCallback
from constants import CLASS_NO


def train(args):
    tf.config.experimental_run_functions_eagerly(True)

    model_name, batch_size = args.model, args.batch_size

    model_class = common.get_model_class(model_name)

    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    res_dir = os.path.join("results", model_name, dt_string)
    os.makedirs(res_dir)

    model: tf.keras.Model = model_class.get_model()

    # if 'pydot' in sys.modules and 'graphviz' in sys.modules:
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(res_dir, 'model.png'), show_shapes=True)
    except:
        pass

    test_input_fn, _ = model_class.get_input_fn_and_steps_per_epoch('test', 8)

    callbacks = []

    if args.reduce_on_plateu:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(patience=args.reduce_on_plateu_patience,
                                                 factor=args.reduce_on_plateu_factor, verbose=1,
                                                 min_delta=args.reduce_on_plateu_min_delta, monitor='loss')
        )

    if args.use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                          patience=args.early_stopping_patience,
                                                          min_delta=args.early_stopping_min_delta,
                                                          verbose=1))

    callbacks.extend([
        tf.keras.callbacks.ModelCheckpoint(os.path.join(res_dir, "weights.hdf5"),
                                           save_best_only=True, verbose=1, save_weights_only=True,
                                           monitor='val_mean_intersection_over_union',
                                           mode='max'),
        MyTensorBoardCallback(args, test_input_fn, res_dir, write_graph=False, profile_batch=0)
    ])

    common.save_model_architecture(res_dir, model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.base_lr),
        loss=common.sparse_crossentropy_ignoring_last_label,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy, common.sparse_accuracy_ignoring_last_label,
                 common.mean_intersection_over_union]
    )

    train_input_fn, train_steps_per_epoch = \
        model_class.get_input_fn_and_steps_per_epoch('train', batch_size)
    validation_input_fn, validation_steps_per_epoch = \
        model_class.get_input_fn_and_steps_per_epoch('valid', batch_size)

    class_weight = {idx: 0. if idx == 0 else 1. / float(CLASS_NO) for idx in range(CLASS_NO)}
    model.fit(train_input_fn, steps_per_epoch=train_steps_per_epoch, epochs=args.epochs, callbacks=callbacks,
              validation_data=validation_input_fn, validation_steps=validation_steps_per_epoch,
              validation_freq=1, class_weight=class_weight)


def main():
    parser = argparse.ArgumentParser(description='DL-MAI project #1 (FNN/CNN) training script.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_models = [model_name.split("/")[1].split(".")[0] for model_name in glob("models/*.py") if
                        "__init__.py" not in model_name]
    parser.add_argument('model', choices=available_models, help='Model name')
    parser.add_argument('--comment', type=str, help='Comment to write to TensorBoard')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of batch per training step')
    parser.add_argument('--base-lr', default=0.01, type=float, help='Base learning rate')
    parser.add_argument('--reduce-on-plateu', action='store_true',
                        help='If provided, will reduce learning rate when no improvement')
    parser.add_argument('--reduce-on-plateu-patience', default=3, type=int,
                        help='After how many epochs without improvement learning rate should be decreased')
    parser.add_argument('--reduce-on-plateu-factor', default=0.1, type=float,
                        help='The factor that learning rate should be moultiplied when plateu')
    parser.add_argument('--reduce-on-plateu-min-delta', default=0.005, type=float,
                        help='The value of change that make callback think there is no plateu')
    parser.add_argument('--use-early-stopping', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-min_delta', type=float, default=0.005)
    # parser.add_argument('--dataset-dir', type=str, default='.', help='Path to directory of dataset')
    # parser.add_argument('--weights', type=lambda x: is_valid_file(parser, x),
    #                     help='Path to weights of pretrained model')
    # parser.add_argument('--test', action='store_true',
    #                     help='Whether run training and evaluation test or not (1 epoch, 1 step, 1 example)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
