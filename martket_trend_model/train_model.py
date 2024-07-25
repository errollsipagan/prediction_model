import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv('aapl_raw_data_history.csv')

# Step 2: Explore the Data
print(df.head())  # View the first few rows of the dataset
print(df.info())  # Get information about the dataset

# Step 3: Preprocess the Data
# Example: Encode categorical variables
# Assume 'Trend' is a categorical variable in the dataset
df['Trend'] = df['Trend'].map({'Bull': 0, 'Bear': 1})

# Handle missing values if any (example: fill missing values with the mean)
df.fillna(df.mean(), inplace=True)

# Step 4: Split the Data into Features and Labels
# Assume 'Trend' is the label and the rest are features
X = df.drop('Trend', axis=1)
y = df['Trend']

# Step 5: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Export the Trained Model
model_filename = 'market_trend_prediction_model.joblib'
joblib.dump(model, model_filename)
print(f'Model saved to {model_filename}')

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 9: Make Predictions on New Data
new_data = pd.DataFrame([[1, 1]], columns=['Day_Of_Week', 'Week_Of_Year'])
predictions = model.predict(new_data)
print(predictions)