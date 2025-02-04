import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model

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




model_AI = load_model(f"../images/knowledgeBase/currentAISolution.keras", custom_objects={'PetalsLSTM': PetalsLSTM})

# Load the model
with open("../images/knowledgeBase/currentOlsSolution.pkl", "rb") as file:
    ols_model= pickle.load(file)
    
# Load the training data
test_df = pd.read_csv(f"../images/learningBase/validation/test_data.csv")

# Define independent variables (input features) and dependent variable (label)
X_test= test_df[['Input1', 'Input2', 'Input3', 'Input4', 'Input5']]
y_test = test_df['Label']  # Target column

# Convert each input row into a tuple (sorted, order-independent bin)
def create_bin(row):
    return tuple(sorted(row))  # Sort values within each vector

# Apply binning to create labels for groups
X_test_binned = X_test.copy()
X_test_binned["Bin"] = X_test_binned.apply(create_bin, axis=1)

# Keep only the first occurrence of each unique bin
X_test_unique = X_test_binned.drop_duplicates(subset=["Bin"], keep="first")

# Extract corresponding y_test values
y_test_unique = y_test.loc[X_test_unique.index]


# Sort bins based on numeric values
X_test_unique["Label"] = y_test_unique
X_test_unique["Bin_Sort"] = X_test_unique["Bin"].apply(lambda x: sum(v * (10 ** (len(x) - i - 1)) for i, v in enumerate(x)))
X_test_sorted = X_test_unique.sort_values(by=["Bin_Sort"])
y_test_sorted = y_test_unique.loc[X_test_sorted.index]
y_pred_unique = ols_model.predict(X_test_sorted.drop(columns=["Bin","Bin_Sort", "Label"]))
y_pred_unique_AI = pd.Series(model_AI.predict(X_test_sorted.drop(columns=["Bin", "Bin_Sort", "Label"])))


# Prepare bin labels for X-axis
#bin_labels = X_test_sorted["Bin"].astype(str).tolist()  # Convert to strings

# Plot scatter for actual values
plt.figure(figsize=(20, 10))
plt.scatter(range(len(X_test_sorted)), y_test_sorted, color='blue', label='Actual Values', alpha=0.6)

# Plot line for predicted values
plt.plot(range(len(X_test_sorted)), y_pred_unique, color='red', label='OLS Predictions', linestyle='solid')

# Plot line for predicted values
plt.plot(range(len(X_test_sorted)), y_pred_unique_AI, color='blue', label='AI Predictions', linestyle='solid')

# Convert tuples to short format
bin_labels = ["".join(map(str, bin_tuple)) for bin_tuple in X_test_sorted["Bin"]]

# Show every N-th label
N = max(1, len(bin_labels) // 20)  

plt.xticks(
    ticks=[i for i in range(len(bin_labels)) if i % N == 0],  
    labels=[bin_labels[i] for i in range(len(bin_labels)) if i % N == 0],  
    rotation=45, ha='right', fontsize=8
)

# Add vertical bin labels to X-axis
#plt.xticks(ticks=range(len(bin_labels)), labels=bin_labels, rotation=90, ha='center', fontsize=8)  # Vertical labels

# Define Y-axis ticks from 0.0 to 20.0 with steps of 2
plt.yticks(np.arange(0.0, 20.1, 2.0))  # 20.1 ensures 20.0 is included

# Labels and title
plt.xlabel("Binned Input Data")
plt.ylabel("Label Value")
plt.title("OLS Model Predictions vs. Actual Values (Binned Inputs)")
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('scatterplott_OLS_AI_v1_.pdf')
