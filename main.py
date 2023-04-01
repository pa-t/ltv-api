import boto3
import logging
import joblib
import json
import pandas as pd

from fastapi import FastAPI, status, File, UploadFile
from io import BytesIO

from utils.preprocess import preprocess_data

app = FastAPI()
logger = logging.getLogger()

# Define the S3 bucket and key where the model is stored
BUCKET_NAME = 'your-s3-bucket-name'
MODEL_KEY = 'your/s3/model/key'
SAGEMAKER_ENDPOINT = 'your-sagemaker-endpoint-name'
BATCH_SIZE = 100

# Define the SageMaker runtime client
runtime = boto3.Session(region_name='us-east-1').client('sagemaker-runtime')


def preprocess_predict(dataset: pd.DataFrame):
  # TODO: implement this preprocessing
  return dataset


@app.post("/model/predict-csv", status_code=status.HTTP_200_OK)
def predict_csv(file: UploadFile = File(...)):
  # TODO: do we need to read in which timeframe to call the right model?
  try:
     # read file contents into a BytesIO stream
    contents = file.file.read()
    buffer = BytesIO(contents)
    # convert stream into dataframe
    df = pd.read_csv(buffer)
    # clean up file 
    buffer.close()
    file.file.close()

    # massage the input data to get it ready to use for prediction
    df = preprocess_predict(dataset=df)

    # Convert the preprocessed data to a JSON-formatted string
    payload = df.to_json(orient='records')
    
    # Invoke the SageMaker endpoint to make predictions
    results = []
    for start in range(0, len(payload), BATCH_SIZE):
      end = min(start + BATCH_SIZE, len(payload))
      subset = payload.iloc[start:end].to_csv(sep=",", header=False, index=False)
      response = runtime.invoke_endpoint(
          EndpointName=SAGEMAKER_ENDPOINT,
          ContentType='text/csv',
          Body=subset
      )
      result = response['Body'].read()
      data_list = [float(x) for x in result.decode('utf-8').strip().split('\n')]
      results.append(data_list)

    # output_df = pd.DataFrame(results)

    return {'predictions': results}
  except Exception as e:
    ERR_MSG = f'Exception encountered predicting: {e}'
    logger.error(ERR_MSG)
    return {'status': ERR_MSG}


@app.post("/model/train-csv", status_code=status.HTTP_202_ACCEPTED)
def train_csv(file: UploadFile = File(...)):
  try:
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

    # Load the existing model from S3
    s3 = boto3.resource('s3')
    with BytesIO() as model_file:
      s3.Bucket(BUCKET_NAME).download_fileobj(MODEL_KEY, model_file)
      model_file.seek(0)
      model = joblib.load(model_file)
    
    # TODO: does this work?
    model.fit(df)

    # Save the updated model back to S3
    with BytesIO() as model_file:
      joblib.dump(model, model_file)
      model_file.seek(0)
      s3.Object(BUCKET_NAME, MODEL_KEY).put(Body=model_file.read())

    return {'status': 'Training job submitted successfully'}
  except Exception as e:
    ERR_MSG = f'Exception encountered training model: {e}'
    logger.error(ERR_MSG)
    return {'status': ERR_MSG}