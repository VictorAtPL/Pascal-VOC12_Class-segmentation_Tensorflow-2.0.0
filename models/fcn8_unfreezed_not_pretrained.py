from tensorflow import keras

from models import fcn8_unfreezed


class Model(fcn8_unfreezed.Model):

    def get_model(self, **kwargs) -> keras.Model:
        model = super().get_model(vgg_weights=None)

        return model
