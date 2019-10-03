from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Activation

from constants import CLASS_NO
from models import fcn8


class Model(fcn8.Model):

    def get_model(self, **kwargs) -> keras.Model:
        img_input = Input(shape=(224, 224, 3))

        fcn32 = Conv2D(CLASS_NO, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(img_input)

        output = Activation('softmax')(fcn32)

        model = keras.Model(img_input, output)

        return model
