from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Create FastAPI app
app = FastAPI()

# Load city_day model
with open("city_day_rf_model.pkl", "rb") as f:
    city_day_model = pickle.load(f)

# Load station_day model
station_day_model = joblib.load("station_day_rf_model.pkl")

# Define input features using Pydantic
class AQIInput(BaseModel):
    City: int
    PM2_5: float
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float
    Benzene: float
    Toluene: float
    Xylene: float
    AQI_Bucket: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Air Quality Prediction API!"}

# City Day Prediction endpoint
@app.post("/predict_city_day/")
def predict_city_day(data: AQIInput):
    try:
        features = np.array([[
            data.City, data.PM2_5, data.PM10, data.NO, data.NO2, data.NOx,
            data.NH3, data.CO, data.SO2, data.O3,
            data.Benzene, data.Toluene, data.Xylene,
            data.AQI_Bucket
        ]])
        prediction = city_day_model.predict(features)
        return {"predicted_AQI": round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e)}

# Station Day Prediction endpoint
@app.post("/predict_station_day/")
def predict_station_day(data: AQIInput):
    try:
        features = np.array([[
            data.City, data.PM2_5, data.PM10, data.NO, data.NO2, data.NOx,
            data.NH3, data.CO, data.SO2, data.O3,
            data.Benzene, data.Toluene, data.Xylene,
            data.AQI_Bucket
        ]])
        prediction = station_day_model.predict(features)
        return {"predicted_AQI": round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e)}

# Serve static HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route for serving the webpage
@app.get("/web", response_class=FileResponse)
def get_webpage():
    return FileResponse("static/index.html")
