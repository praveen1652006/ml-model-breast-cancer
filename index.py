from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize the app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/xgboost_model.pkl')  # Make sure this path is correct
scaler = joblib.load('models/scaler.pkl')  # Same here, correct the path if needed


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_list = []
        for feature in ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
                        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry',
                        'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error',
                        'area_error', 'smoothness_error', 'compactness_error', 'concavity_error',
                        'concave_points_error', 'symmetry_error', 'fractal_dimension_error', 'worst_radius',
                        'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness',
                        'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']:
            feature_value = request.form.get(feature, '')
            if feature_value:
                feature_list.append(float(feature_value))
            else:
                raise ValueError(f"Missing value for {feature}")

        final_features = np.array(feature_list).reshape(1, -1)
        final_features_scaled = scaler.transform(final_features)

        prediction = model.predict(final_features_scaled)

        if prediction[0] == 1:
            result = 'Benign Tumor (Non-cancerous)'
        else:
            result = 'Malignant Tumor (Cancerous)'
    except Exception as e:
        result = f'Error occurred: {e}'

    return render_template('index.html', prediction_text=f'Prediction: {result}')


if __name__ == "__main__":
    app.run(debug=True)
