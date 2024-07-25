import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the Trained Model
model_filename = 'gender_prediction_model.joblib'
model = joblib.load(model_filename)
print('Model loaded from', model_filename)

# Example data (replace with actual data or load from file)
# Generate synthetic data
X, y = np.array([[170, 10], [165, 12], [180, 8], [175, 9], [160, 14], [185, 6],
                 [155, 15], [190, 5], [150, 16], [200, 7]]), np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0])

df = pd.DataFrame(X, columns=['Height', 'Hair_Length'])
df['Gender'] = y

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plotting function
def plot_decision_boundary(model, X, y, feature_names):
    # Create a mesh grid for plotting decision boundary
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the class for each point in the mesh grid
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='Viridis', opacity=0.3, showscale=False))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='Viridis', line=dict(width=1)), name='Data Points'))
    fig.update_layout(title='Decision Boundary and Data Points', xaxis_title=feature_names[0], yaxis_title=feature_names[1])
    
    return fig

# Plot decision boundary
fig = plot_decision_boundary(model, X_scaled, y, feature_names=['Height', 'Hair_Length'])
fig.write_html('decision_boundary.html')
fig.show()
