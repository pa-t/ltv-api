import logging
from fastapi import FastAPI, status
from typing import List

from domain.schemas import RecordSchema, PredictionSchema

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