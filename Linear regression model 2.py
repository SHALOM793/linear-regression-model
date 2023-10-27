# Load the saved model
model = pickle.load(open('road_accident_severity_model.pkl', 'rb'))

# Create a dictionary of the hypothetical independent variables
hypothetical_independent_variables = {
    'road_type': 'urban',
    'weather_condition': 'clear',
    'traffic_volume': 'heavy',
    'speed_limit': 50
}

# Convert the dictionary to a NumPy array
hypothetical_independent_variables = np.array([list(hypothetical_independent_variables.values())])

# Predict the accident severity
accident_severity_prediction = model.predict(hypothetical_independent_variables)

print('Predicted accident severity:', accident_severity_prediction)
