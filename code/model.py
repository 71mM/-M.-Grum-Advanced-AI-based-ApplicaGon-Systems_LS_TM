import tensorflow as tf
from tensorflow.keras import layers, Model

class PetalsLSTM(Model):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(PetalsLSTM, self).__init__()
        self.lstm = layers.LSTM(hidden_size, return_sequences=True, recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(1)

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        lstm_out = self.lstm(x)
        output = self.fc(lstm_out)
        return tf.squeeze(output, axis=-1)
