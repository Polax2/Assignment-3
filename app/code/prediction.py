import joblib
import pandas as pd
import numpy as np

inference_old = joblib.load("car_price_model.pkl")
inference_new = joblib.load("car_price_model2.pkl")
inference_range = joblib.load("car_price_model3.pkl")

def predict_price_old(brand, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):

    data = pd.DataFrame({
        "brand": [brand],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    data = pd.get_dummies( data, columns=["brand", "fuel", "seller_type", "transmission"])
    data = data.reindex(columns=inference_old["columns"], fill_value=0)
    data["year"] = inference_old["scaler_year"].transform(data[["year"]])
    data["km_driven"] = inference_old["scaler_km"].transform(data[["km_driven"]])
    data["mileage"] = inference_old["scaler_mileage"].transform(data[["mileage"]])
    data["engine"] = inference_old["scaler_engine"].transform(data[["engine"]])
    data["max_power"] = inference_old["scaler_power"].transform(data[["max_power"]])
    data["seats"] = inference_old["scaler_seats"].transform(data[["seats"]])

    price = inference_old["model"].predict(data)
    price = np.exp(price)

    return price[0]

def predict_price_new(brand, year, km_driven, fuel, seller_type,
                      transmission, owner, mileage, engine, max_power, seats):

    data = pd.DataFrame({
        "brand": [brand],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    #Encoding fuel and transmission in the same way as during A2 preprocessing
    fuel_mapping = {
        "Diesel": 0,
        "Petrol": 1}

    transmission_mapping = {
        "Automatic": 0,
        "Manual": 1}

    data["fuel"] = data["fuel"].map(fuel_mapping)
    data["transmission"] = data["transmission"].map(transmission_mapping)
    data = pd.get_dummies(data, columns=["brand", "fuel", "seller_type", "transmission"])
    data = data.reindex(columns=inference_new["columns"], fill_value=0)
    data["year"] = inference_new["scaler_year"].transform(data[["year"]])
    data["km_driven"] = inference_new["scaler_km"].transform(data[["km_driven"]])
    data["mileage"] = inference_new["scaler_mileage"].transform(data[["mileage"]])
    data["engine"] = inference_new["scaler_engine"].transform(data[["engine"]])
    data["max_power"] = inference_new["scaler_power"].transform(data[["max_power"]])
    data["seats"] = inference_new["scaler_seats"].transform(data[["seats"]])

    #Converting the data to NumPy and adding the intercept required by the custom model
    data = data.to_numpy().astype(float)
    intercept = np.ones((data.shape[0], 1))
    data = np.concatenate((intercept, data), axis=1)

    #Adding weights
    theta = np.asarray(inference_new["theta"], dtype=float)
    price = data @ theta
    price = np.exp(price)
    return price[0]

def predict_price_range(brand, year, km_driven, fuel, seller_type,
                        transmission, owner, mileage, engine, max_power, seats):

    data = pd.DataFrame({
        "brand": [brand],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    data = pd.get_dummies(data, columns=["brand", "fuel", "seller_type", "transmission"])
    data = data.reindex(columns=inference_range["columns"], fill_value=0)

    data["year"] = inference_range["scaler_year"].transform(data[["year"]])
    data["km_driven"] = inference_range["scaler_km"].transform(data[["km_driven"]])
    data["mileage"] = inference_range["scaler_mileage"].transform(data[["mileage"]])
    data["engine"] = inference_range["scaler_engine"].transform(data[["engine"]])
    data["max_power"] = inference_range["scaler_power"].transform(data[["max_power"]])
    data["seats"] = inference_range["scaler_seats"].transform(data[["seats"]])

    data = data.to_numpy().astype(float)
    intercept = np.ones((data.shape[0], 1))
    data = np.concatenate((intercept, data), axis=1)

    theta = np.asarray(inference_range["theta"], dtype=float)
    z = data @ theta
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=1)[0]

    bins = np.asarray(inference_range["price_bins"], dtype=float)
    return bins[predicted_class], bins[predicted_class + 1]
