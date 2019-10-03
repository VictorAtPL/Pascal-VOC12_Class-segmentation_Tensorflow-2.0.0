from tensorflow import keras

from models import fcn8


class Model(fcn8.Model):

    def get_model(self, **kwargs) -> keras.Model:

        pretrained_weights = kwargs['vgg_weights'] if 'vgg_weights' in kwargs else 'imagenet'

        model = super().get_model(vgg_weights=pretrained_weights)

        # Unfreeze layers
        for layer in model.layers:
            if "block" in layer.name:
                layer.trainable = True

        return model
