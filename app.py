import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the pre-trained model
model = pickle.load(open("best_rf_model.pkl", "rb"))

# Define the encoding dictionaries
austim_encoded = {'Yes': 1, 'No': 0}
country_encoded = {
    'United States': 0,
    'Jordan': 1,
    'United Kingdom': 2,
    'Brazil': 3,
    'New Zealand': 4,
    'United Arab Emirates': 5,
    'India': 6,
    'Kazakhstan': 7,
    'Iraq': 8,
    'France': 9,
    'Malaysia': 10,
    'Viet Nam': 11,
    'Egypt': 12,
    'Netherlands': 13,
    'Canada': 14,
    'Australia': 15,
    'Afghanistan': 16,
    'Oman': 17,
    'Italy': 18,
    'Bahamas': 19,
    'Saudi Arabia': 20,
    'Aruba': 21,
    'Russia': 22,
    'Bolivia': 23,
    'Azerbaijan': 24,
    'Armenia': 25,
    'Serbia': 26,
    'Ethiopia': 27,
    'Sri Lanka': 28,
    'Sweden': 29,
    'Austria': 30,
    'Iceland': 31,
    'Hong Kong': 32,
    'China': 33,
    'Germany': 34,
    'Spain': 35,
    'Tonga': 36,
    'Pakistan': 37,
    'Iran': 38,
    'Argentina': 39,
    'South Africa': 40,
    'Japan': 41,
    'Mexico': 42,
    'Ireland': 43,
    'Nicaragua': 44,
    'Sierra Leone': 45,
    'American Samoa': 46,
    'Ukraine': 47,
    'Czech Republic': 48,
    'Niger': 49,
    'Romania': 50,
    'Cyprus': 51,
    'Belgium': 52,
    'Burundi': 53,
    'Bangladesh': 54
}
gender_encoded = {'Male': 1, 'Female': 0}
jaundice_encoded = {'Yes': 1, 'No': 0}

# Create Flask app
app = Flask(__name__)

# Configure the static folder
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    form_data = request.form.to_dict()

    # Convert features to encoded values
    country = form_data['country']
    gender = form_data['gender']
    austim = form_data['austim']

    encoded_country = country_encoded.get(country, -1)
    encoded_gender = gender_encoded.get(gender, -1)
    encoded_austim = austim_encoded.get(austim, -1)

    # Prepare features for prediction
    features = np.array([
        float(form_data['A1_Score']),
        float(form_data['A2_Score']),
        float(form_data['A3_Score']),
        encoded_austim,
        float(form_data['age']),
        encoded_country
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        return render_template("index.html", prediction_text="<span style='color: orange;'>The person is autistic.</span>")
    else:
        return render_template("index.html", prediction_text="<span style='color: orange;'>The person is not autistic.</span>")

if __name__ == "__main__":
    app.run(debug=True)
