from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model and label encoder
model = joblib.load('size_predictor_model.pkl')
encoder = joblib.load('gender_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        gender = request.form['gender']

        # Validate input ranges
        if not (2<= weight <= 68 and 40 <= height <= 180):
            return render_template(
                'index.html',
                message="Weight and height must be within valid ranges (Weight: 7-50 kg, Height: 65-140 cm).",
                weight=weight,
                height=height,
                gender=gender
            )

        # Encode gender
        gender_encoded = encoder.transform([gender])[0]

        # Make prediction
        prediction = model.predict([[weight, height, gender_encoded]])

        return render_template(
            'index.html',
            prediction=f"Predicted Size: {prediction[0]}",
            weight=weight,
            height=height,
            gender=gender
        )
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        return render_template(
            'index.html',
            message="An error occurred during prediction. Please try again.",
            weight=request.form.get('weight'),
            height=request.form.get('height'),
            gender=request.form.get('gender')
        )

if __name__ == "__main__":
    app.run(debug=True)
