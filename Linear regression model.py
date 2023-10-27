import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('road_accident_severity.csv')

# Define the dependent and independent variables
dependent_variable = 'accident_severity'
independent_variables = ['road_type', 'weather_condition', 'traffic_volume', 'speed_limit']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[independent_variables], df[dependent_variable], test_size=0.25, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
r2_score = model.score(X_test, y_test)

print('R-squared score:', r2_score)

# Save the model for future use
import pickle

pickle.dump(model, open('road_accident_severity_model.pkl', 'wb'))
