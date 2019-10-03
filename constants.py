INPUT_WIDTH = 224
INPUT_HEIGHT = INPUT_WIDTH

FORMAT_LEADING_ZEROS = ':03d'
TFRECORDS_FORMAT_PATTERN = '{}-{' + FORMAT_LEADING_ZEROS + '}-of-{' + FORMAT_LEADING_ZEROS + '}.tfrecords'

CLASS_NO = 21  # background + classes, without void
DATASET_DIRECTORY_PATH = "VOCdevkit/VOC2012"
TFRECORDS_SAVE_PATH = "tfrecords"

LABEL_IDS_TO_NAME = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv/monitor",
    255: "void"
}

LABEL_NAMES_TO_ID = {value: key for key, value in LABEL_IDS_TO_NAME.items()}
VGG_MEAN = [103.939, 116.779, 123.68]