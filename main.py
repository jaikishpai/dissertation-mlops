import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Creating FastAPI instance
from sklearn.neighbors import LocalOutlierFactor

app = FastAPI(
    title="Local Outlier Factor API",
    description="### API for outlier detection of Energy Intensity Data 🚀 <br/>",
    version="0.1",
    contact={
        "name": "Jaikish Pai",
        "Id": "2020SC04666"
    }
)


# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    portfolio_asset_id: int
    country_encoded: float
    sub_region_name_encoded: float
    development_status_encoded: float
    property_type_code_encoded: float
    asset_size_m2: float
    en_int_kwh_m2: float


model_name = "Local Outlier Factor Algorithm"
version = 0.1

# setting contamination factor for outlier detection model
contamination = 0.001

model_local_outlier_factor = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=contamination)
dataset = pd.read_csv("dataset .csv")
dataset.drop("en_abs", inplace=True, axis=1)
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
        data.en_int_kwh_m2
    ]]
    # predict the class
    class_predict = model_local_outlier_factor.predict(test_data)[0]

    '''
    Returning the Outlier status
    if the Outlier Score is -1 ===> Outlier
    if the Outlier Score is 1 ===> Normal
    '''

    outlier = "outlier" if class_predict.item() == -1 else "normal"

    '''
    JSON Response along with Outlier status and Outlier score
    '''

    return {
        "portfolio_asset_id": data.portfolio_asset_id,
        "country_encoded": data.country_encoded,
        "sub_region_name_encoded": data.sub_region_name_encoded,
        "development_status_encoded": data.development_status_encoded,
        "property_type_code_encoded": data.property_type_code_encoded,
        "asset_size_m2": data.asset_size_m2,
        "en_int_kwh_m2": data.en_int_kwh_m2,
        "outlier_score": class_predict.item(),
        "outlier": outlier
    }
