Here's the complete README in a single Markdown block. Replace the `{...}` placeholders with the actual numbers from your notebooks, or just keep them as is for now—you can update after a quick re-run.

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

## Workflow Overview

1. **Exploratory Data Analysis**
   - Distribution of the target, correlation heatmap, scatter plots of key features vs target.
   - Geographical visualization showing higher prices along the coast.
   - Boxplots to identify outliers (especially in `AveRooms`).

2. **Feature Engineering & Preprocessing**
   - Created derived features: `Bedrms_per_Room`, `Rooms_per_Person`, `Households`, `Old_House` (binary).
   - Clipped outliers in `AveRooms`, `AveBedrms`, `AveOccup`, `Population` using IQR limits.
   - Train/test split (80/20, `random_state=42`).
   - Standardised features with `StandardScaler` (fit on training data only).

3. **Baseline Model**
   - Linear Regression: RMSE = {rmse_lr:.4f}, R² = {r2_lr:.4f}

4. **Advanced Models**
   - Decision Tree (`max_depth=10`): RMSE = {rmse_dt:.4f}, R² = {r2_dt:.4f}
   - XGBoost (`n_estimators=200`, `learning_rate=0.1`, `max_depth=6`): RMSE = {rmse_xgb:.4f}, R² = {r2_xgb:.4f}
   - XGBoost gave the lowest error and highest R².

5. **Model Interpretation with SHAP**
   - Summary plot of global feature importance (median income is the strongest driver).
   - Dependence plot showing interaction between `MedInc` and `HouseAge`.
   - Waterfall plot explaining individual predictions.

6. **Deployment (FastAPI)**
   - Saved trained XGBoost model and scaler with `joblib`.
   - Built a `POST /predict` endpoint that receives raw house features, applies the same preprocessing (feature creation, clipping, scaling), and returns the predicted price in USD.
   - Interactive API documentation available at `/docs`.

## Setup Instructions

1. **Clone and enter the project**
   ```bash
   git clone https://github.com/NavidAhmadiii/california-house-price-prediction.git
   cd california-house-price-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Explore the notebooks**
   ```bash
   jupyter notebook
   ```

5. **Run the API**
   ```bash
   cd api
   uvicorn app:app --reload
   ```
   Open `http://127.0.0.1:8000/docs` to test the endpoint.

## Example API Call

**Request** (`POST /predict`)
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

- Best model: **XGBoost** with R² ≈ {r2_xgb:.2f}, RMSE ≈ ${rmse_xgb*100000:,.0f}.
- Top predictor: `MedInc` (median income), followed by geographic features and housing density.
- The preprocessing pipeline ensures no data leakage, and all experiments are reproducible (`random_state=42`).
