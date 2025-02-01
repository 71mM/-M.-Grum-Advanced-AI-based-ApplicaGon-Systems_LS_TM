import tensorflow as tf
from model import PetalsLSTM

def test_model(model, test_sequence, model_path, max_sequence_length):

    model = tf.keras.models.load_model(model_path)
    padded_sequence = test_sequence + [0] * (max_sequence_length - len(test_sequence))
    input_tensor = tf.convert_to_tensor([padded_sequence], dtype=tf.float32)
    prediction = model(input_tensor).numpy().squeeze(0)

    return prediction[:len(test_sequence)]

model = PetalsLSTM(input_size=1, hidden_size=32, num_layers=1)
test_sequence = [3, 5, 1, 2, 6, 5, 3, 3]
predicted_petals = test_model(model, test_sequence,"/content/best_model_20250127_140300.pkl")
true_petals = 14

predicted_last = round(predicted_petals[-1], 2)
true_last = true_petals[-1]

