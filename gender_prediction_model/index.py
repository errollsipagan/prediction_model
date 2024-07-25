import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 3: Create the Dataset
data = {
    'Height': [170, 165, 180, 175, 160, 185, 155, 190, 150, 200],
    'Hair_Length': [10, 12, 8, 9, 14, 6, 15, 5, 16, 7],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)

# Step 4: Preprocess the Data
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Step 5: Split the Data into Training and Testing Sets
X = df[['Height', 'Hair_Length']]
y = df['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 9: Make Predictions on New Data
new_data = np.array([[175, 10], [160, 15]])
predictions = model.predict(new_data)
print(predictions)
