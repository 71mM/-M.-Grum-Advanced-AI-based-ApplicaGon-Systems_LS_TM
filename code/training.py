from model import PetalsLSTM
import tensorflow as tf
import tensorflow.keras as keras
from datetime import datetime
import matplotlib.pyplot as plt

""" Hier muss das dataset eingebundenwerden
max_sequence_length = 50
input_sequences, target_sequences = generate_training_data()
inputs, targets = preprocess_data(input_sequences, target_sequences, max_sequence_length)
"""
epochs = 1000
patience = 25
best_loss = float('inf')
best_model_path = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"

# Annahme: PetalsLSTM ist ein benutzerdefiniertes Modell, das bereits erstellt wurde
model = PetalsLSTM(input_size=1, hidden_size=32, num_layers=1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Callback zur Berechnung des Testverlusts nach jeder Epoche
class TestLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Berechne den Testverlust nach jeder Epoche
        test_loss = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        self.test_losses.append(test_loss)
        print(f"Epoch {epoch+1}: Test Loss = {test_loss}")

# Testdaten (für die Berechnung des Testverlusts)
test_data = (X_test, y_test)

# Callbacks für Early Stopping, Modell-Speicherung und Testverlust-Überwachung
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='loss')
test_loss_callback = TestLossCallback(test_data)

# Trainiere das Modell
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, test_loss_callback],
    verbose=1
)

# Plot erstellen: Train Loss vs Test Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(test_loss_callback.test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Test Loss')

# Plot als PDF speichern
pdf_path = f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
plt.savefig(pdf_path)
print(f"Plot gespeichert unter: {pdf_path}")
