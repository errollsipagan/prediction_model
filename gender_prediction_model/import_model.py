import numpy as np
import joblib
model_filename = 'gender_prediction_model.joblib'
# Making Predictions on New Data
def predict_new_data(model, new_data):
    predictions = model.predict(new_data)
    return predictions

# Load the Trained Model
loaded_model = joblib.load(model_filename)
print('Model loaded from', model_filename)

# Make Predictions with the Loaded Model
new_data = np.array([[175, 10], [170, 10]])
predictions = predict_new_data(loaded_model, new_data)
print(predictions)
