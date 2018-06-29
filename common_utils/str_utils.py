from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.constraints import maxnorm, nonneg, unitnorm, min_max_norm


class StrUtils:

    @staticmethod
    def str_write(obj):
        if isinstance(obj, (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)):
            return obj.__class__.__name__ + ' : ' + str(obj.get_config())

        if isinstance(obj, (maxnorm, nonneg, unitnorm, min_max_norm)):
            return obj.__class__.__name__ + ' : ' + str(obj.get_config())

        return str(obj)
