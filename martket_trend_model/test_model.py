import pandas as pd
import joblib
model_filename = 'market_trend_prediction_model.joblib'
# Making Predictions on New Data
def predict_new_data(model, new_data):
    predictions = model.predict(new_data)
    return predictions

# Load the Trained Model
loaded_model = joblib.load(model_filename)
print('Model loaded from', model_filename)

# Make Predictions with the Loaded Model
new_data = pd.DataFrame([[1, 30],[2, 30],[3, 30],[4, 30],[5, 30]], columns=['Day_Of_Week', 'Week_Of_Year'])
predictions = predict_new_data(loaded_model, new_data)
print(predictions)
