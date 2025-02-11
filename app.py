from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the dataset and model
df = pd.read_csv('Updated_Crop_recommendation.csv')
model = pickle.load(open('model.pkl', 'rb'))
minmaxscaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Mapping from crop numeric IDs to names
crop_dict = {
    1: "rice", 2: "wheat", 3: "maize", 4: "jute", 5: "cotton",
    6: "coconut", 7: "papaya", 8: "orange", 9: "apple",
    10: "muskmelon", 11: "watermelon", 12: "grapes", 13: "mango",
    14: "banana", 15: "pomegranate", 16: "lentil", 17: "blackgram",
    18: "mungbean", 19: "mothbeans", 20: "pigeonpeas", 21: "kidneybeans",
    22: "chickpea", 23: "coffee"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting input data from the form
        input_data = {
            'Nitrogen': int(request.form['Nitrogen']),
            'Phosphorus': int(request.form['Phosphorus']),
            'Potassium': int(request.form['Potassium']),
            'Temperature': float(request.form['Temperature']),
            'Humidity': float(request.form['Humidity']),
            'Ph': float(request.form['Ph']),
            'Rainfall': float(request.form['Rainfall']),
            'Area': float(request.form['Area'])
        }

        # Validation for positive values and Ph range
        for key, value in input_data.items():
            if value < 0:
                raise ValueError(f"{key} should be a positive value.")

        if not (0 <= input_data['Ph'] <= 14):
            raise ValueError("Ph value must be between 0 and 14.")

        # Preparing data for model prediction
        features = np.array([list(input_data.values())[:-1]]).reshape(1, -1)
        scaled_features = minmaxscaler.transform(features)
        prediction = model.predict(scaled_features)
        crop_name = crop_dict[prediction[0]]

        # Retrieve crop details from the dataset
        crop_data = df[df['label'] == crop_name].iloc[0]
        average_price = crop_data['price']
        yield_per_acre_kg = crop_data['yield_per_acre_kg']
        expenditure_per_acre = crop_data['expenditure_per_acre']

        # Calculate profit
        revenue = average_price * yield_per_acre_kg * input_data['Area']
        total_expenditure = expenditure_per_acre * input_data['Area']
        profit = revenue - total_expenditure

        result = {
            'crop': crop_name,
            'revenue': f"₹{revenue:.2f}",
            'expenditure': f"₹{total_expenditure:.2f}",
            'profit': f"₹{profit:.2f}"
        }
        return render_template('index.html', result=result)
    except ValueError as ve:
        # Catch validation errors
        return render_template('index.html', error=str(ve))
    except Exception as e:
        # Catch any other errors
        return render_template('index.html', error="An error occurred: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
