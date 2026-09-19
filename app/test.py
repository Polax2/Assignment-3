import numpy as np
import joblib
inference = joblib.load("code/car_price_model3.pkl")

def test_model_input_shape():
    n_features = len(inference["columns"]) + 1
    model_input = np.zeros((1, n_features))

    assert model_input.shape == (1, 47)

def test_model_output_shape():
    n_features = len(inference["columns"]) + 1
    model_input = np.zeros((1, n_features))

    theta = np.asarray(inference["theta"], dtype=float)
    z = model_input @ theta
    prediction = np.argmax(z, axis=1)

    assert prediction.shape == (1,)