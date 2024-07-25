import pandas as pd
import plotly.express as px
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

# Create a scatter plot using Plotly
fig = px.scatter(df, x='Day_Of_Week', y='Week_Of_Year', color='Trend', 
                 labels={'Trend': 'Trend (0=Bull, 1=Bear)'}, 
                 title='Day_Of_Week vs Week_Of_Year')

# Save the plot to an HTML file
fig.write_html('market_trend_scatter_plot.html')

# Display the plot in the notebook
fig.show()
