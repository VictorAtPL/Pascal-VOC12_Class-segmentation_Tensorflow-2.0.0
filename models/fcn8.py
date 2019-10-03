from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Add, Activation

from AbstractModel import AbstractModel
from common import load_sets_count, get_input_fn_and_steps_per_epoch, \
    pascal_voc2012_segmentation_annotated_parser, pascal_voc2012_segmentation_not_annotated_parser
from constants import CLASS_NO, TFRECORDS_SAVE_PATH


class Model(AbstractModel):

    sets_count = load_sets_count()

    def get_model(self, **kwargs) -> keras.Model:
        img_input = Input(shape=(224, 224, 3))

        pretrained_weights = kwargs['vgg_weights'] if 'vgg_weights' in kwargs else 'imagenet'

        # Encoder - Load VGG16
        encoder = VGG16(weights=pretrained_weights, input_tensor=img_input,
                        include_top=False, pooling=None)

        # Encoder -  Freeze layers
        if pretrained_weights:
            for layer in encoder.layers:
                if "block" in layer.name:
                    layer.trainable = False

        # Encoder -  Get intermediate VGG16 layers output
        pool3 = encoder.get_layer('block3_pool').output
        pool4 = encoder.get_layer('block4_pool').output
        pool5 = encoder.get_layer('block5_pool').output

        # Encoder - add two convolution layers
        conv6 = (Conv2D(filters=4096, kernel_size=(7, 7), activation='relu', padding='same', name='block6_conv1'))(
            pool5)
        conv7 = (Conv2D(filters=4096, kernel_size=(1, 1), activation='relu', padding='same', name='block7_conv1'))(
            conv6)

        # Decoder - fcn32 - not used, but written for the learning purpose
        # conv8 = Conv2D(filters=N_CLASSES, kernel_size=(1, 1), activation='relu', padding='same', name='block8_conv1')(conv7)
        # fcn32 = Conv2DTranspose(N_CLASSES, kernel_size=(1, 1), strides=(32, 32), padding='valid', name='block8_deconv1)(conv8)

        # Decoder - fcn16 - not used, but written for the learning purpose
        # conv9 = Conv2D(filters=N_CLASSES, kernel_size=(1, 1), activation='relu', padding='same', name='block9_conv1')(pool4)
        # deconv1x = Conv2DTranspose(filters=N_CLASSES, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same', name='block9_deconv1')(conv7)
        # add1 = Add(name='block9_add1')([conv9, deconv1])
        # fcn16 = Conv2DTranspose(N_CLASSES, kernel_size=(1, 1), strides=(16, 16), padding='valid', name='block9_deconv2')(add1)

        # Decoder - fcn32
        conv10 = Conv2D(filters=CLASS_NO, kernel_size=(1, 1), activation='relu', padding='same', name='block10_conv1')(
            pool3)
        conv11 = Conv2D(filters=CLASS_NO, kernel_size=(1, 1), activation='relu', padding='same', name='block10_conv2')(
            pool4)
        deconv2 = Conv2DTranspose(filters=CLASS_NO, kernel_size=(1, 1), strides=(2, 2), activation='relu',
                                  padding='same', name='block10_deconv1', use_bias=False)(conv11)
        deconv3 = Conv2DTranspose(filters=CLASS_NO, kernel_size=(1, 1), strides=(4, 4), activation='relu',
                                  padding='same', name='block10_deconv2', use_bias=False)(conv7)
        add2 = Add(name='block10_add2')([conv10, deconv2, deconv3])
        fcn32 = Conv2DTranspose(CLASS_NO, kernel_size=(1, 1), strides=(8, 8), padding='valid', name='block10_deconv3',
                                use_bias=False)(add2)

        output = Activation('softmax')(fcn32)

        model = keras.Model(img_input, output)

        return model

    @classmethod
    def get_input_fn_and_steps_per_epoch(cls, set_name, batch_size=None):
        parser_fn = pascal_voc2012_segmentation_annotated_parser

        if 'test' in set_name:
            parser_fn = pascal_voc2012_segmentation_not_annotated_parser

        return get_input_fn_and_steps_per_epoch(set_name, parser_fn, TFRECORDS_SAVE_PATH,
                                                batch_size, cls.sets_count)
