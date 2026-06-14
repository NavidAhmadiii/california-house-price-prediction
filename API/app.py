from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


app = FastAPI()

model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')


class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int
    YearRemodAdd: int
    GarageArea: int
    TotRmsAbvGrd: int
    Fireplaces: int

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }


@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_dict = features.dict()
    df = pd.DataFrame([input_dict])

    df['Bedrms_per_Room'] = df['AveBedrms'] / df['AveRooms']
    df['Rooms_per_Person'] = df['AveRooms'] / df['AveOccup']
    df['Households'] = df['Population'] / df['AveOccup']
    df['Old_House'] = (df['HouseAge'] > 50).astype(int)

    clip_limits = {
        'AveRooms': (2.0, 12.0),
        'AveBedrms': (1.0, 3.0),
        'AveOccup': (1.0, 6.0),
        'Population': (3.0, 5000.0)
    }

    for col, (lower, upper) in clip_limits.items():
        df[col] = df[col].clip(lower=lower, upper=upper)

    scaled_input = scaler.transform(df)
    prediction = model.predict(scaled_input)[0]
    price_in_dollars = prediction * 100_000
    return {
        "predicted_median_house_value": round(price_in_dollars, 2),
        "unit": "USD"
    }

# مسیر ریشه برای تست


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running. Go to /docs for Swagger UI."}
