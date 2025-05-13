import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def train_and_save_models(X_train, y_train, models_dir="models"):

    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Prepare features (remove reference columns)
    if 'year_original' in X_train.columns:
        X_train_features = X_train.drop(columns=['year_original'])
    else:
        X_train_features = X_train
    
    # Initialize and train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_features, y_train)
    
    # Initialize and train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_features, y_train)
    
    # Initialize and train a Polynomial Regression model (degree 2)
    poly_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )
    poly_model.fit(X_train_features, y_train)
    
    # Save models
    joblib.dump(rf_model, os.path.join(models_dir, "model_random_forest.pkl"))
    joblib.dump(lr_model, os.path.join(models_dir, "model_linear_regression.pkl"))
    joblib.dump(poly_model, os.path.join(models_dir, "model_polynomial.pkl"))
    
    return {
        "random_forest": rf_model,
        "linear_regression": lr_model,
        "polynomial": poly_model
    }

def evaluate_model(model, X_test, y_test):
\
    # Prepare features (remove reference columns)
    if 'year_original' in X_test.columns:
        X_test_features = X_test.drop(columns=['year_original'])
    else:
        X_test_features = X_test
    
    # Make predictions
    predictions = model.predict(X_test_features)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Calculate MAPE (Mean Absolute Percentage Error) with handling for zero values
    # To avoid division by zero
    mask = y_test != 0
    y_true_safe = y_test[mask]
    y_pred_safe = predictions[mask]
    
    if len(y_true_safe) > 0:
        mape = mean_absolute_percentage_error(y_true_safe, y_pred_safe)
    else:
        mape = np.nan
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "predictions": predictions
    }