from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate, BatchNormalization

from constants import CLASS_NO
from models import fcn8


class Model(fcn8.Model):
    n_filters = 16
    kernel_size = 3
    dropout = 0.25

    def get_model(self, **kwargs) -> keras.Model:
        img_input = Input(shape=(224, 224, 3))

        # Start with 224x224x3. Apply 3x3x16 Convolution, padding same and 2x2 Pooling. New dimensions:
        # 112x112x16
        c1 = self._conv2d_block(n_filters=self.n_filters * 1, kernel_size=self.kernel_size, input_tensor=img_input)
        p1 = MaxPooling2D((2, 2))(c1)
        d1 = Dropout(self.dropout)(p1)

        # 112x112x16. Apply 3x3x32 Convolution, padding same and 2x2 Pooling. New dimensions:
        # 56x56x32
        c2 = self._conv2d_block(n_filters=self.n_filters * 2, kernel_size=self.kernel_size, input_tensor=d1)
        p2 = MaxPooling2D((2, 2))(c2)
        d2 = Dropout(self.dropout)(p2)

        # 56x56x32. Apply 3x3x64 Convolution, padding same and 2x2 Pooling. New dimensions:
        # 28x28x64
        c3 = self._conv2d_block(n_filters=self.n_filters * 4, kernel_size=self.kernel_size, input_tensor=d2)
        p3 = MaxPooling2D((2, 2))(c3)
        d3 = Dropout(self.dropout)(p3)

        # 28x28x64. Apply 3x3x128 Convolution, padding same and 2x2 Pooling. New dimensions:
        # 14x14x128
        c4 = self._conv2d_block(n_filters=self.n_filters * 8, kernel_size=self.kernel_size, input_tensor=d3)
        p4 = MaxPooling2D((2, 2))(c4)
        d4 = Dropout(self.dropout)(p4)

        # 14x14x128. Apply 3x3x256 Convolution, padding same. New dimensions: 14x14x256
        c5 = self._conv2d_block(n_filters=self.n_filters * 16, kernel_size=self.kernel_size, input_tensor=d4)

        # Upsampling part starts here
        # Start with dimensions 14x14x256
        u6 = Conv2DTranspose(self.n_filters * 8, kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(2, 2), padding='same')(c5)
        u6 = Concatenate()([u6, c4])
        d6 = Dropout(self.dropout)(u6)
        c6 = self._conv2d_block(self.n_filters * 8, kernel_size=3, input_tensor=d6)

        u7 = Conv2DTranspose(self.n_filters * 4, kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(2, 2), padding='same')(c6)
        u7 = Concatenate()([u7, c3])
        d7 = Dropout(self.dropout)(u7)
        c7 = self._conv2d_block(self.n_filters * 4, kernel_size=3, input_tensor=d7)

        u8 = Conv2DTranspose(self.n_filters * 2, kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(2, 2), padding='same')(c7)
        u8 = Concatenate()([u8, c2])
        d8 = Dropout(self.dropout)(u8)
        c8 = self._conv2d_block(self.n_filters * 2, kernel_size=3, input_tensor=d8)

        u9 = Conv2DTranspose(self.n_filters * 1, kernel_size=(self.kernel_size, self.kernel_size),
                             strides=(2, 2), padding='same')(c8)
        u9 = Concatenate()([u9, c1])
        d9 = Dropout(self.dropout)(u9)
        c9 = self._conv2d_block(self.n_filters * 1, kernel_size=3, input_tensor=d9)

        # Apply 1x1 convolution
        outputs = Conv2DTranspose(CLASS_NO, (1, 1), activation='softmax')(c9)

        model = keras.Model(inputs=[img_input], outputs=[outputs])

        return model

    @staticmethod
    def _conv2d_block(n_filters, kernel_size=3, batch_norm=True, input_tensor=None):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                   padding='same')(input_tensor)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                   padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)

        return x
