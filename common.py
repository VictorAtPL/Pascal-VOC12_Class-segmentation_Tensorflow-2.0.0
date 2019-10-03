import math
import os
import pickle
from glob import glob
from importlib import import_module
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from PIL import Image

from AbstractModel import AbstractModel
from constants import TFRECORDS_SAVE_PATH, TFRECORDS_FORMAT_PATTERN, FORMAT_LEADING_ZEROS, INPUT_WIDTH, INPUT_HEIGHT, \
    VGG_MEAN


def make_input_local(filenames, parser_fn, shuffle=False, repeat=False, cache=False, shuffle_buffer_size=300):
    def input_fn(params):
        batch_size = params["batch_size"]

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if cache:
            dataset = dataset.cache()

        if repeat:
            dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

        if batch_size:
            dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    return input_fn


def get_input_fn_and_steps_per_epoch(set_name, parser_fn, tfrecords_path, batch_size, sets_count):
    assert set_name in ('train', 'valid', 'test')

    steps_per_epoch = None
    if batch_size:
        steps_per_epoch = math.ceil(sets_count[set_name] / batch_size)

    input_fn = make_input_local(
        glob(os.path.join(tfrecords_path, TFRECORDS_FORMAT_PATTERN.replace(FORMAT_LEADING_ZEROS, '').format(set_name, '*', '*'))),
        parser_fn,
        shuffle=True if set_name == 'train' else False,
        repeat=True if set_name == 'train' else False,
        cache=True if set_name == 'train' else False)({'batch_size': batch_size})

    return input_fn, steps_per_epoch


def pascal_voc2012_segmentation_parser(serialized_example, annotated=True):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "filename": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "annotation": tf.io.FixedLenFeature([], tf.string),
        })

    image_bytes = features['image']
    image = tf.io.decode_raw(image_bytes, tf.uint8)
    image = tf.reshape(image, [INPUT_HEIGHT, INPUT_WIDTH, 3])
    image = tf.cast(image, tf.float32)

    if not annotated:
        return image

    annotation_bytes = features['annotation']
    annotation = tf.io.decode_raw(annotation_bytes, tf.uint8)
    annotation = tf.reshape(annotation, [INPUT_HEIGHT, INPUT_WIDTH])

    return image, annotation


def pascal_voc2012_segmentation_not_annotated_parser(serialized_example):
    return pascal_voc2012_segmentation_parser(serialized_example, annotated=False)


def pascal_voc2012_segmentation_annotated_parser(serialized_example):
    return pascal_voc2012_segmentation_parser(serialized_example)


def save_model_architecture(res_dir, model):
    # save model config to json
    model_json = model.to_json()
    with open(os.path.join(res_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)


def get_model_class(model_name: str) -> AbstractModel:
    return import_module("models.{}".format(model_name)).Model()


def load_pickle_file(filename):
    with tf.io.gfile.GFile(os.path.join(TFRECORDS_SAVE_PATH, '../', filename), mode='rb') as f:
        return pickle.load(f)


def load_sets_count():
    sets_count = load_pickle_file('sets_count.pickle')

    return sets_count


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


def blend_from_3d_sparse(image, annotation, cmap, alpha=0.35):
    annotation[annotation == 21] = 255

    if image.dtype == np.float32:
        image = (image * 255.0).astype(np.uint8)

    image = Image.fromarray(image)
    annotation = Image.fromarray(cmap[annotation])
    return np.array(Image.blend(image, annotation, alpha))


def blend_from_3d_one_hots(plt, annotation, cmap, alpha=0.5):
    annotation = np.argmax(annotation, axis=-1)
    return blend_from_3d_sparse(plt, annotation, cmap, alpha)


@tf.function
def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    """
    Source: https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/utils.py
    """
    y_true = tf.cast(y_true, tf.uint8)
    nb_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(y_true, nb_classes+1)[:, :, :, :-1]
    return K.categorical_crossentropy(y_true, y_pred)


@tf.function
def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    """
    Source: https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/utils.py
    """
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.dtypes.cast(K.flatten(y_true), tf.int64)
    legal_labels = ~K.equal(y_true, nb_classes)
    return K.sum(tf.cast(legal_labels & K.equal(y_true, K.argmax(y_pred, axis=-1)), tf.float32)) / K.sum(tf.cast(legal_labels, tf.float32))


@tf.function
def mean_intersection_over_union(y_true, y_pred):
    """
    Source: https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/master/utils.py
    """
    nb_classes = K.int_shape(y_pred)[-1]
    intersection_over_union = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(nb_classes):  # exclude last label (void)
        true_labels = K.equal(y_true[:, :, :], i)
        pred_labels = K.equal(pred_pixels, i)
        intersection = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1) > 0
        intersection_over_unions = K.sum(intersection, axis=1) / K.sum(union, axis=1)

        intersection_over_union.append(K.mean(intersection_over_unions[legal_batches]))
    intersection_over_union = tf.stack(intersection_over_union)
    legal_labels = ~tf.math.is_nan(intersection_over_union)
    intersection_over_union = intersection_over_union[legal_labels]
    return K.mean(intersection_over_union)
