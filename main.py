import logging
import pandas as pd
from fastapi import FastAPI, status, File, UploadFile
from io import BytesIO
from typing import List

from domain.schemas import RecordSchema, PredictionSchema
from utils.preprocess import preprocess_data

import random # TODO: delete this

app = FastAPI()
logger = logging.getLogger()



# TODO: where do we need to call the preprocess function?


@app.post("/model/predict")
def predict(records: List[RecordSchema]) -> List[PredictionSchema]:
  # call the predict endpoint from sagemaker
  logger.info(f"records: {records}")

  response = [
    PredictionSchema(
      customer_id=record.customer_id,
      predicted_ltv=random.random(0, 30),
      time_range=record.time_range
    ) for record in records
  ]

  # return response from predicting
  return response


@app.post("/model/train", status_code=status.HTTP_204_NO_CONTENT)
# TODO: what schema should we make this take? cols from table?
def train(records: List[RecordSchema]):
  # call the train endpoint from sagemaker
  logger.info(f"records to train on: {records}")

  return


@app.post("/model/train-csv", status_code=status.HTTP_204_NO_CONTENT)
def train_csv(file: UploadFile = File(...)):
  # read file contents into a BytesIO stream
  contents = file.file.read()
  buffer = BytesIO(contents)
  # convert stream into dataframe
  df = pd.read_csv(buffer)
  # clean up file 
  buffer.close()
  file.file.close()

  # TODO: do we want to do any validation that the columns are correct?

  df = preprocess_data(dataset=df)

  # call the training endpoint

  return