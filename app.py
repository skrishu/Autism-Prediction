import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the pre-trained model
model = pickle.load(open("rf_classifier_selected.pkl", "rb"))

# Define the country and ethnicity encoding dictionaries
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

ethnicity_encoded = {
    'White-European': 0,
    'Middle Eastern': 1,
    'Pasifika': 2,
    'Black': 3,
    'Others': 4,
    'Hispanic': 5,
    'Asian': 6,
    'Turkish': 7,
    'South Asian': 8,
    'Latino': 9
}

# Create Flask app
app = Flask(__name__)

#Configure the static folder
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
    ethnicity = form_data['ethnicity']
    
    encoded_country = country_encoded.get(country, -1)  # -1 indicates unknown country
    encoded_ethnicity = ethnicity_encoded.get(ethnicity, -1)  # -1 indicates unknown ethnicity
    
    if encoded_country == -1 or encoded_ethnicity == -1:
        return render_template("index.html", prediction_text="Invalid country or ethnicity.")
    
    # Prepare features for prediction
    float_features = [float(form_data[feature]) for feature in form_data if feature not in ['country', 'ethnicity']]
    features = [np.array(float_features + [encoded_country, encoded_ethnicity])]
    
    # Make prediction
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text="<span style='color: orange;'>The result is {}</span>".format(prediction))



if __name__ == "__main__":
    app.run(debug=True)
