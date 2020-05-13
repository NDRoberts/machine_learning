import tensorflow as tf
import kerastuner as kt


class DiaperModel(kt.HyperModel):

    def __init__(self):
        pass

    def build(self, hyparams):
        model = tf.keras.Sequential()
        for k in range(hyparams.Int('layers', 2, 10)):
            model.add(tf.keras.layers.Dense(
                units=hyparams.Int('units_' + str(k), min_value=3, max_value=21, step=3),
                activation=hyparams.Choice('activation_' + str(k), ['relu', 'tanh', 'linear'])))
        model.compile(
            optimizer='adadelta',
            loss='mean_squared_error',
            metrics='mean_squared_error')
        return model


if __name__ == '__main__':
    dmod = DiaperModel()
    print("I made", dmod)