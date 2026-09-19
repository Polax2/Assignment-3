# Car Price Prediction- Car's Price Reveal Party

## About the project

The project is built based on the provided Car Price dataset. It contains the full data preprocessing, modelling and then, based on the chosen model, a simple web application was built and deployed.

## Machine Learning

The project was developed throughout three stages.

In the first version, four Machine Learning models were tested: Linear Regression, Random Forest, Decision Tree and SVM. Their performance was evaluated using MSE, RMSE and R² metrics. Linear Regression was chosen as the final model.

In stage 2, the Linear Regression implementation was extended and different combinations of optimization methods, initialization methods, momentum, learning rates and regularization were tested and compared using MSE and R². The best model was used as the Advanced model in the web application.

In stage 3, the selling price was divided into 4 classes with Multinomial Logistic Regression. The models were evaluated using Accuracy, Precision, Recall, F1, Macro F1 and Weighted F1. Ridge with lambda 0.005 was selected as the final model based on the Mean Weighted F1 score. This stage's model is used by the PRICE RANGE option in the web application.

## Repository

The repository contains the Jupyter Notebook with the full data investigation, preprocessing, model training and evaluation. It also contains the `app` folder with all files needed to run the web application.

## Running the application

The application is deployed on the AIT server and is available through the following webpage:
https://web-st127173.ml.brain.cs.ait.ac.th

## Web Application

The web application contains three main pages: the home page, the instruction page and the prediction page. From the home page the user can navigate either to the instructions or directly to the predictor.

On the prediction page, some fields are required and some can be left empty.
Required fields:

- Brand
- Year
- Kilometers driven
- Transmission
- Engine
- Max power

Optional fields:

- Fuel
- Seller type
- Owner
- Mileage
- Seats

If an optional field is left empty, the application fills the missing value using the imputation values calculated from the training data or the most common mode of the featire withing the training set.
The application also contains basic input validation. Required fields cannot be left empty and negative values for numerical features are rejected before the prediction is made.

## Used tools

The project was developed using Python, Pandas, Scikit-learn and Dash. Docker was used to containerize the final web application.

## What is new?

The prediction page has been extended with a new model selection functionality. The user can now choose between three prediction models:

- **Standard**- uses the basic Linear Regression model from the first version of the application
- **Advanced**- uses the improved Linear Regression model developed in Assignment 2
- **PRICE RANGE**- uses the Multinomial Logistic Regression model developed in Assignment 3. Instead of predicting one exact price, this model predicts the price range in which the car price lies

The user can now choose whether to obtain an exact price prediction using one of the previous models or a predicted price range using the new classification model.
