import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
import pandas as pd

class PetalsLSTM(Model):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, **kwargs):
        super(PetalsLSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = layers.LSTM(hidden_size, recurrent_initializer='glorot_uniform')

        self.fc = layers.Dense(1)

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        lstm_out = self.lstm(x)
        output = self.fc(lstm_out)
        return tf.squeeze(output, axis=-1)

    def get_config(self):
        config = super(PetalsLSTM, self).get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers
        })
        return config

    @classmethod
    def from_config(cls, config):

        return cls(**config)



model = load_model('currentSolution.keras', custom_objects={'PetalsLSTM': PetalsLSTM})

activation_data = pd.read_csv(f'../images/activationBase/activation_data.csv')
label = activation_data['Label']
activation_data_input = activation_data.drop(columns=['Label'])

prediction = model.predict(activation_data_input)
print("Input:")
print(activation_data_input)
print("------------------------------------------")
print("Prediction: ", prediction[0])
print("actual label: ", label[0])




