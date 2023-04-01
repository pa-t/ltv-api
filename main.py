import boto3
import logging
import joblib
import json
import pandas as pd

from fastapi import FastAPI, Response, File, UploadFile
from io import BytesIO

from utils.preprocess import preprocess_data, preprocess_predict

app = FastAPI()
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Define the S3 bucket and key where the model is stored
BUCKET_NAME = 'variantcustomerdata'
# TODO: PROGRAMATICALLY PICK WHICH MODEL
MODEL_KEY = 'ltv-365-unfiltered/sagemaker-automl-candidates/model/WeightedEnsemble-L3-FULL-t1/model.tar.gz'
SAGEMAKER_ENDPOINT = 'variant-ltv-365'
BATCH_SIZE = 100

# Define the SageMaker runtime client
runtime = boto3.Session(region_name='us-east-1').client('sagemaker-runtime')


@app.post("/model/predict")
def predict(file: UploadFile = File(...)):
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

    # drop target column
    # TODO: programattically handle this
    df = df.drop(columns=['total_spent_365'])
    
    # Invoke the SageMaker endpoint to make predictions
    results = []
    for start in range(0, len(df), BATCH_SIZE):
      end = min(start + BATCH_SIZE, len(df))
      subset = df.iloc[start:end].to_csv(sep=",", header=False, index=False)
      logger.info(f"Predict batch complete: {round((end/len(df))*100, 2)}%")
      response = runtime.invoke_endpoint(
          EndpointName=SAGEMAKER_ENDPOINT,
          ContentType='text/csv',
          Body=subset
      )
      result = response['Body'].read()
      data_list = [float(x) for x in result.decode('utf-8').strip().split('\n')]
      results.extend(data_list)

    # TODO: we need to associate the predicted LTV with the customer ID
    return Response(
      content=json.dumps({'predictions': results}),
      media_type='application/json'
    )
  except Exception as e:
    ERR_MSG = f'Exception encountered predicting: {e}'
    logger.error(ERR_MSG)
    return Response(content=json.dumps({'status': ERR_MSG}), media_type='application/json', status_code=500)


@app.post("/model/train")
def train(file: UploadFile = File(...)):
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

    return Response(
      content={
        'status': 'Training job submitted successfully'
      },
      media_type='application/json'
    )
  except Exception as e:
    ERR_MSG = f'Exception encountered training model: {e}'
    logger.error(ERR_MSG)
    return Response(content=json.dumps({'status': ERR_MSG}), media_type='application/json', status_code=500)