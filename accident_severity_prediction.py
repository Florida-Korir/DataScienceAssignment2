import json
import numpy as np

def load_model(json_path):
    """Loads the trained linear regression model from a JSON file."""
    with open(json_path, "r") as file:
        model_data = json.load(file)
    return model_data

def predict_accident_severity(input_data, model_data):
    """Predicts accident severity using the loaded model coefficients."""
    coefficients = np.array(model_data["coefficients"])
    intercept = model_data["intercept"]
    
    input_array = np.array(input_data)
    prediction = np.dot(input_array, coefficients) + intercept
    return prediction

if __name__ == "__main__":
    # Load the trained model
    model_path = "accident_severity_model.json"  # Ensure this file is in the same directory
    model = load_model(model_path)
    
    # Example input (values should match the number of features in the dataset)
    sample_input = [2, 1, 0, 3, 60, 1]  # Example values for features
    
    # Make a prediction
    severity_prediction = predict_accident_severity(sample_input, model)
    print(f"Predicted Accident Severity: {severity_prediction}")
