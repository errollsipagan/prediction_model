import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 3: Create the Dataset
data = {
    'Height': [170, 165, 180, 175, 160, 185, 155, 190, 150, 200],
    'Hair_Length': [10, 12, 8, 9, 14, 6, 15, 5, 16, 7],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)

# Step 4: Preprocess the Data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Split the Data into Features and Labels
X = df[['Height', 'Hair_Length']]
y = df['Gender']

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Phase
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Train the Model
model = train_model(X_train, y_train)

# Export the Trained Model
model_filename = 'gender_prediction_model.joblib'
joblib.dump(model, model_filename)
print(f'Model saved to {model_filename}')

# Evaluation Phase
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Evaluate the Model
accuracy, y_pred = evaluate_model(model, X_test, y_test)
print(f'Accuracy: {accuracy}')

# Making Predictions on New Data
def predict_new_data(model, new_data):
    predictions = model.predict(new_data)
    return predictions

# New Data for Prediction
new_data = np.array([[175, 10], [160, 15]])
predictions = predict_new_data(model, new_data)
print(predictions)
