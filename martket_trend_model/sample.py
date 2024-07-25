import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 3: Create the Dataset
data = {
    'Day_Of_Week': [1, 2, 3, 5, 1, 3, 4, 2, 3, 1],
    'Week_Of_Year': [1, 4, 8, 9, 14, 6, 15, 5, 16, 7],
    'Trend': ['Bull', 'Bear', 'Bull', 'Bull', 'Bear', 'Bull', 'Bear', 'Bull', 'Bear', 'Bull']
}

df = pd.DataFrame(data)

# Step 4: Preprocess the Data
df['Trend'] = df['Trend'].map({'Bull': 0, 'Bear': 1})

# Step 5: Split the Data into Training and Testing Sets
X = df[['Day_Of_Week', 'Week_Of_Year']]
y = df['Trend']

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
new_data = pd.DataFrame([[1, 1]], columns=['Day_Of_Week', 'Week_Of_Year'])
predictions = model.predict(new_data)
print(predictions)
