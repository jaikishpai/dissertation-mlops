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
class RequestBody(BaseModel):
    country_encoded: float
    sub_region_name_encoded: float
    property_type_code_encoded: float
    development_status_encoded: float
    asset_size_m2: float
    en_int_kwh_m2: float


# setting contamination factor for outlier detection model
contamination = 0.001

model_local_outlier_factor = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=contamination)


@app.post('/outliers')
def outliers(data: request_body):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
