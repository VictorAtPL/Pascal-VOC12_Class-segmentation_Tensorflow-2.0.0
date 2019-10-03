import pickle

import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

from common import load_pickle_file
from constants import INPUT_HEIGHT, INPUT_WIDTH, CLASS_NO, DATASET_DIRECTORY_PATH, TFRECORDS_SAVE_PATH, \
    TFRECORDS_FORMAT_PATTERN


def generate_tf_records_for_dataset(dataset_trainval_files, test_set_files, dataset_segmentation_image_files_path,
                                    dataset_segmentation_annotation_files_path, output_path, split_ratio=0.9):
    if tf.io.gfile.exists(output_path):
        tf.io.gfile.rmtree(output_path)
    tf.io.gfile.mkdir(output_path)

    np.random.seed(0)
    random.shuffle(dataset_trainval_files)
    samples_no = len(dataset_trainval_files)
    train_examples_no = int(split_ratio * samples_no)
    val_examples_no = samples_no - train_examples_no

    train_set_files = dataset_trainval_files[:train_examples_no]
    val_set_files = dataset_trainval_files[-val_examples_no:]

    sets_files = [(train_set_files, 'train'), (val_set_files, 'valid'), (test_set_files, 'test')]

    for set_files, set_name in tqdm(sets_files, position=0):
        generate_tf_records_for_set(set_files, set_name, dataset_segmentation_image_files_path,
                                    dataset_segmentation_annotation_files_path, output_path)


def generate_tf_records_for_set(set_files, set_name, dataset_segmentation_image_files_path,
                                dataset_segmentation_annotation_files_path, output_path):
    set_files_no = len(set_files)

    part_files_arr = np.array_split(set_files, np.ceil(set_files_no / 2000))

    for part_no, part_arr in tqdm(enumerate(part_files_arr), position=1, total=len(part_files_arr)):
        generate_part_of_tf_records_for_set(part_no=part_no,
                                            part_files=part_arr,
                                            set_name=set_name,
                                            no_of_parts=len(part_files_arr),
                                            dataset_segmentation_image_files_path=dataset_segmentation_image_files_path,
                                            dataset_segmentation_annotation_files_path=dataset_segmentation_annotation_files_path,
                                            output_path=output_path)

    write_sets_length(os.path.join(TFRECORDS_SAVE_PATH, '../'), set_name, set_files_no)


def generate_part_of_tf_records_for_set(part_no, part_files, set_name, no_of_parts,
                                        dataset_segmentation_image_files_path,
                                        dataset_segmentation_annotation_files_path, output_path):
    with tf.io.TFRecordWriter(
            os.path.join(output_path, TFRECORDS_FORMAT_PATTERN.format(set_name, part_no + 1, no_of_parts))) as writer:
        for file in tqdm(part_files, position=2):
            segmentation_image_path = os.path.join(dataset_segmentation_image_files_path, file + ".jpg")

            image = Image.open(segmentation_image_path)
            image = image.resize((INPUT_HEIGHT, INPUT_WIDTH))
            image = np.array(image)

            if 'test' not in set_name:
                segmentation_annotation_path = os.path.join(dataset_segmentation_annotation_files_path, file + ".png")
                annotation = Image.open(segmentation_annotation_path)
                annotation = annotation.resize((INPUT_HEIGHT, INPUT_WIDTH))
                annotation = np.array(annotation, dtype=np.uint8)
                annotation[annotation == 255] = CLASS_NO
            else:
                annotation = np.zeros(image.shape[:2])

            example = convert_image_and_annotation_to_tf_example(file, image, annotation)
            writer.write(example.SerializeToString())


def convert_image_and_annotation_to_tf_example(filename, x, y):
    assert (x.shape[0] == y.shape[0] and x.shape[1] == y.shape[
        1]), "Dimensions of image and annotation image must be equal."

    return tf.train.Example(features=tf.train.Features(
        feature={'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.tostring()])),
                 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tostring()])),
                 'annotation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tostring()]))}))


def write_sets_length(path, set_name, set_files_no):
    local_sum_sets_split = {set_name: set_files_no}

    if tf.io.gfile.exists(os.path.join(path, "sets_count.pickle")):
        local_sum_sets_split = load_pickle_file("sets_count.pickle")

        local_sum_sets_split[set_name] = set_files_no

    with tf.io.gfile.GFile(os.path.join(path, "sets_count.pickle"), mode='wb') as f:
        pickle.dump(local_sum_sets_split, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    dataset_segmentation_annotation_files_path = os.path.join(DATASET_DIRECTORY_PATH, "SegmentationClass")
    dataset_segmentation_image_files_path = os.path.join(DATASET_DIRECTORY_PATH, "JPEGImages")
    dataset_trainval_split_file = os.path.join(DATASET_DIRECTORY_PATH, "ImageSets", "Segmentation", "trainval.txt")
    dataset_test_split_file = os.path.join(DATASET_DIRECTORY_PATH, "ImageSets", "Segmentation", "test.txt")

    with open(dataset_trainval_split_file, "r") as f:
        dataset_trainval_files = [file.replace("\n", "") for file in f.readlines()]

    with open(dataset_test_split_file, "r") as f:
        dataset_test_files = [file.replace("\n", "") for file in f.readlines()]

    generate_tf_records_for_dataset(dataset_trainval_files, dataset_test_files, dataset_segmentation_image_files_path,
                                    dataset_segmentation_annotation_files_path, TFRECORDS_SAVE_PATH)


if __name__ == '__main__':
    main()
