import boto3
import logging
import pandas as pd
import sagemaker

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from sagemaker import model_uris
from sagemaker.estimator import Estimator
from starlette.responses import JSONResponse

from domain.enums import ModelTimeFrame
from utils.preprocess import preprocess_data, preprocess_predict

app = FastAPI()
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Define the S3 bucket and key where the model is stored
BUCKET_NAME = 'variantcustomerdata'
# TODO: PROGRAMATICALLY PICK WHICH MODEL
MODEL_KEY = '/model.tar.gz'
BATCH_SIZE = 100

# Define the SageMaker runtime client
runtime = boto3.Session(region_name='us-east-1').client('sagemaker-runtime')


@app.post("/model/predict")
def predict(file: UploadFile = File(...), model_time_frame: ModelTimeFrame = ModelTimeFrame.month):
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

    # drop target column and select which sagemaker model to use
    if model_time_frame.value == "365":
      sagemaker_endpoint = 'variant-ltv-365'
      logger.info("Using year ltv model...")
      df = df.drop(columns=['total_spent_365'])
    elif model_time_frame.value == "90":
      sagemaker_endpoint = 'variant-ltv-90'
      logger.info("Using quarter ltv model...")
      df = df.drop(columns=['total_spent_90'])
    else:
      sagemaker_endpoint = 'variant-ltv-30'
      logger.info("Using month ltv model...")
      df = df.drop(columns=['total_spent_30'])
    
    # Invoke the SageMaker endpoint to make predictions
    results = []
    for start in range(0, len(df), BATCH_SIZE):
      end = min(start + BATCH_SIZE, len(df))
      subset = df.iloc[start:end].to_csv(sep=",", header=False, index=False)
      logger.info(f"Predict batch complete: {round((end/len(df))*100, 2)}%")
      response = runtime.invoke_endpoint(
          EndpointName=sagemaker_endpoint,
          ContentType='text/csv',
          Body=subset
      )
      result = response['Body'].read()
      data_list = [float(x) for x in result.decode('utf-8').strip().split('\n')]
      results.extend(data_list)
    
    # construct dataframe with predicted ltv married to customer id
    results_df = pd.DataFrame({'customer_id': df.customer_id, 'predicted_ltv': results})

    return JSONResponse(
      content={'predictions': results_df.to_dict(orient='records')}
    )
  except Exception as e:
    ERR_MSG = f'Exception encountered predicting: {e}'
    logger.error(ERR_MSG)
    return JSONResponse(content={'status': ERR_MSG}, status_code=500)


@app.post("/model/train")
# TODO: make this default to month
def train(file: UploadFile = File(...), model_time_frame: ModelTimeFrame = ModelTimeFrame.year):
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
    df = preprocess_data(dataset=df, target_width=int(model_time_frame.value))

    # upload preprocessed df to bucket
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    # TODO: add uuid to name of training data
    boto3.client('s3').put_object(Body=csv_buffer.getvalue(), Bucket=BUCKET_NAME, Key=f"training-data/train-data-{model_time_frame}.csv")

    # define training job params
    training_data_path = f's3://{BUCKET_NAME}/training-data/train-data-{model_time_frame}.csv'
    output_path = f's3://{BUCKET_NAME}/output'
    instance_type = 'ml.m4.2xlarge'
    algorithm_image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.4.3-cpu-py38-ubuntu20.04'

    # Set the starting point for the training job
    model_name = 'Experiment-1679959638859-best-model'
    endpoint_name = 'dev-break-this'
    starting_point = f'sagemaker://{model_name}-{endpoint_name}'

    # Create the estimator object
    estimator = Estimator(
      image_uri=algorithm_image,
      instance_type=instance_type,
      role='arn:aws:iam::558967991329:role/CFN-SM-IM-Lambda-Catalog-SageMakerExecutionRole-JF5STS1HP7Y4',
      input_mode='File',
      output_path=output_path,
      sagemaker_session=sagemaker.Session(),
      base_job_name='api-triggered-training-job'
    )

    import pdb;pdb.set_trace()

    # Start the training job
    estimator.fit(inputs=training_data_path, starting_point=starting_point)


    return JSONResponse(
      content={
        'status': 'Training job submitted successfully'
      }
    )
  except Exception as e:
    ERR_MSG = f'Exception encountered training model: {e}'
    logger.error(ERR_MSG)
    return JSONResponse(content={'status': ERR_MSG}, status_code=500)