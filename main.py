import pandas as pd
from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

# Creating FastAPI instance
from sklearn.neighbors import LocalOutlierFactor

app = FastAPI()


# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    portfolio_asset_id: int
    country_encoded: float
    sub_region_name_encoded: float
    development_status_encoded: float
    property_type_code_encoded: float
    asset_size_m2: float
    en_abs: float
    en_int_kwh_m2: float


model_name = "LOF"
version = 1.0

# setting contamination factor for outlier detection model
contamination = 0.001

model_local_outlier_factor = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=contamination)
dataset = pd.read_csv("dataset.csv")
model_local_outlier_factor.fit(dataset.values)


@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.post('/outliers')
def outliers(data: request_body):
    test_data = [[
        data.portfolio_asset_id,
        data.country_encoded,
        data.sub_region_name_encoded,
        data.development_status_encoded,
        data.property_type_code_encoded,
        data.asset_size_m2,
        data.en_abs,
        data.en_int_kwh_m2
    ]]
    # predict the class
    class_predict = model_local_outlier_factor.predict(test_data)[0]

    # return the result
    return {class_predict.item()}

