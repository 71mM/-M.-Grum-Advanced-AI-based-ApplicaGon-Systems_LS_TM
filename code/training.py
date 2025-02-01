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
patience = 25  # Anzahl Epochen ohne Verbesserung vor Abbruch
best_loss = float('inf')  # Anfangswert für den besten Verlust
best_model_path = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"  # Dynamischer Pfad

model = PetalsLSTM(input_size=1, hidden_size=32, num_layers=1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Callbacks für Early Stopping und Modell-Speicherung
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_loss')

history = model.fit(
    train_data, train_labels,
    validation_data=(test_data, test_labels),
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Plot erstellen
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Validation Loss')

# Plot als PDF speichern
pdf_path = f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
plt.savefig(pdf_path)
print(f"Plot gespeichert unter: {pdf_path}")