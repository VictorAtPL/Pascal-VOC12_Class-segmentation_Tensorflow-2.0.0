from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation

from AbstractModel import AbstractModel
from common import load_sets_count, get_input_fn_and_steps_per_epoch, pascal_voc2012_segmentation_annotated_parser, \
    pascal_voc2012_segmentation_not_annotated_parser
from constants import CLASS_NO, INPUT_WIDTH, INPUT_HEIGHT, TFRECORDS_SAVE_PATH
from models import fcn8


class Model(AbstractModel):

    sets_count = load_sets_count()

    def get_model(self, **kwargs) -> keras.Model:
        # img_input = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

        # Encoder - Load VGG16
        # input_tensor = img_input,
        encoder = VGG16(weights='imagenet', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3),
                        include_top=False, pooling=None)

        # Encoder -  Get intermediate VGG16 layers output
        pool2 = encoder.get_layer('block2_pool').output

        # Encoder
        conv6 = (Conv2D(filters=1024, kernel_size=(7, 7), activation='relu', padding='same', name='block6_conv1'))(
            pool2)

        conv7 = (Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', padding='same', name='block7_conv1'))(
            conv6)

        # Decoder
        fcn32 = Conv2DTranspose(CLASS_NO, kernel_size=(7, 7), strides=(4, 4), padding='same', name='block10_deconv3',
                                use_bias=False)(conv7)

        output = Activation('softmax')(fcn32)

        model = keras.Model(encoder.input, output)

        return model

    @classmethod
    def get_input_fn_and_steps_per_epoch(cls, set_name, batch_size=None):
        parser_fn = pascal_voc2012_segmentation_annotated_parser

        if 'test' in set_name:
            parser_fn = pascal_voc2012_segmentation_not_annotated_parser

        return get_input_fn_and_steps_per_epoch(set_name, parser_fn, TFRECORDS_SAVE_PATH,
                                                batch_size, cls.sets_count)
