from abc import ABC, abstractmethod
from tensorflow import keras


class AbstractModel(ABC):
    @abstractmethod
    def get_model(self, **kwargs) -> keras.Model:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_input_fn_and_steps_per_epoch(cls, set_name, batch_size=None):
        raise NotImplementedError
