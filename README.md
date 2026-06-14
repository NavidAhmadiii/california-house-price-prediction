```markdown
# California House Price Prediction

An end-to-end machine learning project that predicts median house values in California districts using 1990 census data. The pipeline covers data exploration, feature engineering, model training, interpretation, and a FastAPI prediction service.

## Dataset
- **Source:** `fetch_california_housing` (scikit-learn)
- **Samples:** 20,640  
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude  
- **Target:** Median house value (in $100,000s)

## Project Structure
```
california-house-price-prediction/
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   └── 02_feature_engineering.ipynb
├── api/
│   ├── model/
│   │   ├── xgboost_model.pkl
│   │   └── scaler.pkl
│   └── app.py
├── .gitignore
├── README.md
└── requirements.txt
```

## What Was Done

### 1. Exploratory Data Analysis
- Visualized target distribution (right‑skewed) and feature correlations.
- Scatter plots revealed a strong positive link between median income and house value.
- Geographic plot showed higher prices along the California coast.
- Boxplots detected extreme outliers in `AveRooms` and `AveOccup`.

### 2. Feature Engineering & Preprocessing
- Created four new features: `Bedrms_per_Room`, `Rooms_per_Person`, `Households`, `Old_House`.
- Clipped outliers in `AveRooms`, `AveBedrms`, `AveOccup`, `Population` using IQR limits.
- Split data into 80% training / 20% test (`random_state=42`).
- Standardized all features with `StandardScaler` (fitted on training data only).

### 3. Baseline Model
- **Linear Regression** – RMSE = 0.73 (≈ $73,000), R² = 0.61

### 4. Advanced Models
- **Decision Tree** (`max_depth=10`) – RMSE = 0.59, R² = 0.67  
- **XGBoost** (`n_estimators=200`, `learning_rate=0.1`, `max_depth=6`) – RMSE = 0.47, R² = 0.83  

XGBoost clearly outperformed the others.

### 5. Model Interpretation with SHAP
- Global importance: median income (`MedInc`) is by far the strongest predictor.
- Dependence plots showed interactions between income and house age.
- Waterfall plots explained individual predictions in dollar terms.

### 6. Deployment (FastAPI)
- Saved the trained XGBoost model and scaler with `joblib`.
- Built a `POST /predict` endpoint that applies the same preprocessing steps and returns the predicted price in USD.
- Interactive documentation at `/docs` via Swagger UI.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/NavidAhmadiii/california-house-price-prediction.git
   cd california-house-price-prediction
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the API**
   ```bash
   cd api
   uvicorn app:app --reload
   ```
   Visit `http://127.0.0.1:8000/docs` to test the prediction.

## Example API Request

**`POST /predict`**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322.0,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Response**
```json
{
  "predicted_median_house_value": 452600.0,
  "unit": "USD"
}
```

## Results Summary
- Best model: **XGBoost** with **R² ≈ 0.83** and **RMSE ≈ $47,000**.
- Median income (`MedInc`) is the most influential feature.
- The preprocessing pipeline avoids data leakage, and the entire workflow is reproducible.
