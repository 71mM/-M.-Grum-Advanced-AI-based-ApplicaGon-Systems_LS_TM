import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from model import PetalsLSTM
import tensorflow as tf
from tensorflow import keras


model_AI = keras.models.load_model(f"C:/Users/lasse/OneDrive/Dokumente/MASTER/Semester WS24/AIBAS/GroupProject/-M.-Grum-Advanced-AI-based-ApplicaGon-Systems_LS_TM-1/best_model_20250201_221433.keras", custom_objects={'PetalsLSTM': PetalsLSTM})

# Load the model
with open("currentOlsSolutionDimensionUpdated.pkl", "rb") as file:
    ols_model= pickle.load(file)
    
# Load the training data
test_df = pd.read_csv(f"test_data.csv")

# Define independent variables (input features) and dependent variable (label)
X_test= test_df[['Input1', 'Input2', 'Input3', 'Input4', 'Input5']]
y_test = test_df['Label']  # Target column

# Make predictions using the loaded model
y_pred_loaded = ols_model.predict(X_test)


## Remove duplicate input rows, keeping only the first occurrence
X_test_unique = X_test.drop_duplicates(keep='first')

# Keep corresponding y_test values (first occurrence only)
y_test_unique = y_test.loc[X_test_unique.index]

# Get predictions for unique X_test values
y_pred_unique = ols_model.predict(X_test_unique)
y_pred_unique_AI = model_AI.predict(X_test_unique)

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


# Get predictions for unique bins
y_pred_unique = ols_model.predict(X_test_unique.drop(columns=["Bin"]))
y_pred_unique_AI = model_AI.predict(X_test_unique.drop(columns=["Bin"]))

# Sort bins based on numeric values
X_test_unique["Bin_Sort"] = X_test_unique["Bin"].apply(lambda x: sum(v * (10 ** (len(x) - i - 1)) for i, v in enumerate(x)))
X_test_sorted = X_test_unique.sort_values(by="Bin_Sort")
y_test_sorted = y_test_unique.loc[X_test_sorted.index]
y_pred_sorted = y_pred_unique.loc[X_test_sorted.index]
y_pred_sorted_AI = y_pred_unique.loc[X_test_sorted.index]



# Prepare bin labels for X-axis
#bin_labels = X_test_sorted["Bin"].astype(str).tolist()  # Convert to strings

# Plot scatter for actual values
plt.figure(figsize=(20, 10))
plt.scatter(range(len(X_test_sorted)), y_test_sorted, color='blue', label='Actual Values', alpha=0.6)

# Plot line for predicted values
plt.plot(range(len(X_test_sorted)), y_pred_sorted, color='red', label='OLS Predictions', linestyle='solid')

# Plot line for predicted values
plt.plot(range(len(X_test_sorted)), y_pred_sorted_AI, color='blue', label='OLS Predictions', linestyle='solid')

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
plt.show()