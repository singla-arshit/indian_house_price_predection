from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load model and mappings
model = joblib.load('xgb_model.pkl')
with open('city_mapping.json', 'r') as f:
    city_mapping = json.load(f)
with open('location_mapping.json', 'r') as f:
    location_mapping = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', cities=city_mapping.keys(), locations=location_mapping.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedrooms = float(request.form['bedrooms'])
        resale = int(request.form['resale'])
        swimming = int(request.form['swimmingpool'])
        parking = int(request.form['carparking'])
        school = int(request.form['school'])
        lift = int(request.form['lift'])
        maintenance = int(request.form['maintenance'])
        location = request.form['location']
        city = request.form['city']
        
        city_encoded = city_mapping.get(city, 0)
        location_encoded = location_mapping.get(location, 0)
        
        input_data = np.array([[area, bedrooms, resale, swimming, parking, school, lift, maintenance, location_encoded, city_encoded]])
        prediction = model.predict(input_data)[0]
        increased_price = prediction * 1.25
        
        return render_template('index.html', 
                              prediction_text=f"Predicted House Price: ₹{int(increased_price):,} Lakhs", 
                              cities=city_mapping.keys(), 
                              locations=location_mapping.keys())
    except Exception as e:
        return render_template('index.html', 
                              prediction_text=f"Error: {str(e)}", 
                              cities=city_mapping.keys(), 
                              locations=location_mapping.keys())

#  API Endpoint
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    try:
        area = float(data['area'])
        bedrooms = float(data['bedrooms'])
        resale = int(data['resale'])
        swimming = int(data['swimmingpool'])
        parking = int(data['carparking'])
        school = int(data['school'])
        lift = int(data['lift'])
        maintenance = int(data['maintenance'])
        location = data['location']
        city = data['city']
        
        city_encoded = city_mapping.get(city, 0)
        location_encoded = location_mapping.get(location, 0)
        
        input_data = np.array([[area, bedrooms, resale, swimming, parking, school, lift, maintenance, location_encoded, city_encoded]])
        prediction = model.predict(input_data)[0]
        increased_price = prediction * 1.25
        
        return jsonify({'predicted_price_lakhs': round(increased_price, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
