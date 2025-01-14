# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime

# 1. Load Data
def load_data(data_path):
    return pd.read_csv(data_path)

# 2. Data Preprocessing
def preprocess_data(data):
    # Strip column names to avoid trailing spaces
    data.columns = data.columns.str.strip()

    # Add missing columns with default values
    if "CompetitionDistance" not in data.columns:
        print("CompetitionDistance column is missing. Adding it with default values.")
        data["CompetitionDistance"] = np.nan

    if "Promo2" not in data.columns:
        print("Promo2 column is missing. Adding it with default values.")
        data["Promo2"] = 0

    if "StoreType" not in data.columns:
        print("StoreType column is missing. Adding it with default values.")
        data["StoreType"] = "a"

    if "Assortment" not in data.columns:
        print("Assortment column is missing. Adding it with default values.")
        data["Assortment"] = "a"

    if "PromoInterval" not in data.columns:
        print("PromoInterval column is missing. Adding it with default values.")
        data["PromoInterval"] = "None"

    # Fill missing values where necessary
    data["CompetitionDistance"].fillna(data["CompetitionDistance"].median(), inplace=True)
    data["Promo2"].fillna(0, inplace=True)
    data["StateHoliday"] = data["StateHoliday"].replace(0, "0")  # Ensure consistency in type

    return data

# 3. Feature Engineering
def engineer_features(data):
    # Extract date-based features
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    
    # Drop unused columns
    data.drop(['Date', 'Customers'], axis=1, inplace=True)
    return data

# 4. Machine Learning Pipeline
def build_pipeline():
    # Separate features by type
    numeric_features = ["CompetitionDistance", "DayOfWeek", "Promo"]
    categorical_features = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

    # Preprocessors
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Pipeline with Random Forest Regressor
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    return pipeline
# 5. Model Training and Evaluation
def train_model(data, target_col):
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build pipeline
    pipeline = build_pipeline()
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    
    return pipeline, mae, rmse

# 6. Serialize Model
def save_model(model, output_dir='models'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = os.path.join(output_dir, f"sales_model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Main function
if __name__ == "__main__":
    # Load and preprocess data
    data_path = "../data/train.csv"  
    data = load_data(data_path)
    data = preprocess_data(data)
    data = engineer_features(data)
    
    # Train and save model
    target_col = "Sales"
    model, mae, rmse = train_model(data, target_col)
    save_model(model)

