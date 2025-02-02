import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load the training data
train_df = pd.read_csv("training_data.csv")

# Define independent variables (input features) and dependent variable (label)
X_train = train_df[['Input1', 'Input2', 'Input3', 'Input4', 'Input5']]  # Feature columns
y_train = train_df['Label']  # Target column

# Add a constant term for intercept in OLS regression
X_train = sm.add_constant(X_train)

# Fit OLS regression model
ols_model = sm.OLS(y_train, X_train).fit()

# Print model summary
print(ols_model.summary())

# Load the test data
test_df = pd.read_csv("test_data.csv")

# Prepare test data for prediction
X_test = test_df[['Input1', 'Input2', 'Input3', 'Input4', 'Input5']]
X_test = sm.add_constant(X_test)  # Add constant term
y_test = test_df['Label']

# Make predictions
y_pred = ols_model.predict(X_test)

# Evaluate model performance using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# --- Saving the model ---
 
# Save the model
with open("currentOlsSolution.pkl", "wb") as file:
    pickle.dump(ols_model, file)

print("OLS model saved as 'ols_model.pkl'.")
