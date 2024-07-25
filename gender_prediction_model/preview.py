import pandas as pd
import plotly.express as px
import joblib

# Load your data (replace with your actual data loading method)
# For example, a simple dataset:
data = {
    'Height': [170, 165, 180, 175, 160, 185, 155, 190, 150, 200],
    'Hair_Length': [10, 12, 8, 9, 14, 6, 15, 5, 16, 7],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
}
df = pd.DataFrame(data)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Create a scatter plot using Plotly
fig = px.scatter(df, x='Height', y='Hair_Length', color='Gender', 
                 labels={'Gender': 'Gender (0=Male, 1=Female)'}, 
                 title='Height vs Hair Length')

# Save the plot to an HTML file
fig.write_html('scatter_plot.html')

# Display the plot in the notebook
fig.show()
