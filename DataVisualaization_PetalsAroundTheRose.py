import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# URL of the GitHub page
GITHUB_URL = "https://github.com/alecGraves/tensorflow-petals-around-the-rose/blob/master/rosepetals.dat"  # Replace with the actual URL

# Convert GitHub page URL to raw file URL if needed
RAW_URL = GITHUB_URL.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

# Fetch the data
response = requests.get(RAW_URL)

if response.status_code == 200:
    content = response.text
    # Extract numbers using regex
    matches = re.findall(r"\[([\d\s]+)\]\s+([\d.]+)", content)
    
    if matches:
        dataset = []
        for match in matches:
            numbers = list(map(int, match[0].split()))
            value = float(match[1])
            dataset.append((numbers, value))
        
        # Print the extracted dataset
        for item in dataset:
            print(item)
    else:
        print("No matching data found.")
else:
    print("Failed to fetch the page. Status Code:", response.status_code)
    
# Convert to DataFrame
df = pd.DataFrame(dataset, columns=['Inputs', 'Label'])

# Expand input lists into separate columns
input_columns = ['Würfel1', 'Würfel2', 'Würfel3', 'Würfel4', 'Würfel5']
df[input_columns] = pd.DataFrame(df['Inputs'].tolist(), index=df.index)
df.drop(columns=['Inputs'], inplace=True)

# --- Descriptive Statistics ---
print("Descriptive Statistics:")
print(df.describe())

# --- Plot histograms of input features ---
plt.figure(figsize=(12, 6))
df[input_columns].hist(figsize=(12, 6), bins=10, layout=(2, 3), edgecolor='black')
plt.suptitle('Histograms of Input Features', fontsize=14)
plt.savefig('histograms_input_features.pdf')
plt.show()

# --- Boxplot of input features ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[input_columns])
plt.title('Boxplot of Input Features')
plt.savefig('boxplot_input_features.pdf')
plt.show()

# --- Histogram of labels ---
plt.figure(figsize=(8, 5))
sns.histplot(df['Label'], bins=10, kde=True, edgecolor='black')
plt.title('Distribution of Labels')
plt.savefig('histogram_labels.pdf') 
plt.show()

# --- Correlation matrix ---
correlation_matrix = df.corr()

# --- Display correlation heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features and Label')
plt.show()

#-----------------------------------------------------------------------------------------
#   Additional outlier detection
#-----------------------------------------------------------------------------------------

# --- IQR Method ---
Q1 = df[input_columns].quantile(0.25)
Q3 = df[input_columns].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = ((df[input_columns] < lower_bound) | (df[input_columns] > upper_bound)).sum()
print("Outliers detected using IQR method:")
print(outliers_iqr)

#-----------------------------------------------------------------------------------------
#   Save th data
#-----------------------------------------------------------------------------------------

# Save the full dataset
df.to_csv("joint_data_collection.csv", index=False)
print("Full dataset saved as 'joint_data_collection.csv'")

# Split the data into training (80%) and testing (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save training data
train_df.to_csv("training_data.csv", index=False)
print("Training dataset saved as 'training_data.csv'")

# Save test data
test_df.to_csv("test_data.csv", index=False)
print("Test dataset saved as 'test_data.csv'")

# Select one random entry from the test set
activation_data = test_df.sample(n=1, random_state=42)

# Save activation data
activation_data.to_csv("activation_data.csv", index=False)
print("One test data entry saved as 'activation_data.csv'")

plt.savefig('correlation_matrix.pdf')
plt.show()

# --- Train-Test Split ---
X = df[input_columns]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split data to CSV files
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Save the entire dataset to a CSV file
df.to_csv('full_dataset.csv', index=False)

print("Train and test data saved as CSV.")

