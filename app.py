from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.losses import MeanSquaredError
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
try:
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load('scaler.pkl')
    logger.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model/scaler: {e}")
    raise e

# Load groundwater dataset (must be yearly data)
try:
    data = pd.read_csv("reshaped_groundwater_rainfall_data.csv")
    data['YEAR'] = data['TIME'].str.extract(r'(\d{4})').astype(int)
    logger.info("✅ Groundwater data loaded.")
except Exception as e:
    logger.error(f"❌ Error loading data: {e}")
    raise e

def validate_input(latitude, longitude, year):
    errors = []
    if not (-90 <= latitude <= 90):
        errors.append("Latitude must be between -90 and 90.")
    if not (-180 <= longitude <= 180):
        errors.append("Longitude must be between -180 and 180.")
    return errors

@app.route('/')
def home():
    return "🌊 Water Level Prediction API - POST to /predict with latitude, longitude, and year."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        logger.info(f"📥 Received data: {request_data}")

        required_fields = ['latitude', 'longitude', 'year']
        if not all(field in request_data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'status': 'error'}), 400

        lat = float(request_data['latitude'])
        lon = float(request_data['longitude'])
        year = int(request_data['year'])

        errors = validate_input(lat, lon, year)
        if errors:
            return jsonify({'errors': errors, 'status': 'error'}), 400

        # Filter historical data
        group = data[(data['LAT'] == lat) & (data['LON'] == lon)].sort_values(by='YEAR')

        if group['GROUNDWATER'].isnull().all() or len(group) < 12:
            return jsonify({'error': 'Insufficient historical yearly groundwater data', 'status': 'error'}), 400

        # Take last 12 years of groundwater data
        last_12_years = group['GROUNDWATER'].fillna(0).values[-12:].reshape(-1, 1)
        scaled_input = scaler.transform(last_12_years).reshape(1, 12, 1)

        # Predict for 8 years: (year - 7) to year
        prediction_years = list(range(year - 7, year + 1))
        future_input = scaled_input.reshape(12, 1)
        future_predictions_scaled = []
        prediction_dict = {}

        for y in prediction_years:
            pred_scaled = model.predict(future_input.reshape(1, 12, 1))[0]
            pred_actual = scaler.inverse_transform(np.array([[pred_scaled[0]]]))[0, 0]
            prediction_dict[y] = round(float(pred_actual), 2)

            # Shift input window
            future_input = np.append(future_input[1:], pred_scaled).reshape(12, 1)

        logger.info(f"✅ Predictions: {prediction_dict}")

        return jsonify({
            'predictions': prediction_dict,
            'latitude': lat,
            'longitude': lon,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"❌ Exception during prediction: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
