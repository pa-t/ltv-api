import logging
import pickle
import pandas as pd

from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from keras.models import load_model, save_model
from starlette.responses import JSONResponse
import tensorflow as tf

from domain.enums import ModelTimeFrame
from domain.exceptions import MissingColumnsException
from utils.preprocess import preprocess_train, get_features, check_columns
from utils.zltv_model import model_predict, zero_inflated_lognormal_loss


app = FastAPI()

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@app.post("/model/predict")
def predict(file: UploadFile = File(...), model_time_frame: ModelTimeFrame=ModelTimeFrame.month):
  try:
     # read file contents into a BytesIO stream
    contents = file.file.read()
    buffer = BytesIO(contents)
    # convert stream into dataframe
    df = pd.read_csv(buffer)
    # clean up file 
    buffer.close()
    file.file.close()

    # convert historical transaction data into customer profile
    df = get_features(dataset=df, target_width=int(model_time_frame.value))

    # drop target column and select which model to use
    if model_time_frame.value == "365":
      model_path = 'models/year/365model.h5'
      logger.info("Using year ltv model...")
    elif model_time_frame.value == "90":
      model_path = 'models/quarter/90model.h5'
      logger.info("Using quarter ltv model...")
    else:
      model_path = 'models/month/30model.h5'
      logger.info("Using month ltv model...")

    # use keras to load in the correct file
    model = load_model(model_path, compile=False, custom_objects={'zero_inflated_lognormal_loss': zero_inflated_lognormal_loss})
    model.compile(loss=zero_inflated_lognormal_loss)

    with open('static_data/feature_map.pkl', 'rb') as f:
      feature_map = pickle.load(f)
    
    # call predict on the model and get the results
    results = model_predict(model, data=df, feature_map=feature_map)
    
    # construct dataframe with predicted ltv married to customer id
    results_df = pd.DataFrame({'customer_id': df.index, 'predicted_ltv': results.ltv_prediction})

    return JSONResponse(
      content={'predictions': results_df.to_dict(orient='records')}
    )
  except Exception as e:
    ERR_MSG = f'Exception encountered predicting: {e}'
    logger.error(ERR_MSG)
    return JSONResponse(content={'status': ERR_MSG}, status_code=500)


@app.post("/model/train")
def train(file: UploadFile = File(...), model_time_frame: ModelTimeFrame=ModelTimeFrame.month):
  try:
    # read file contents into a BytesIO stream
    contents = file.file.read()
    buffer = BytesIO(contents)
    # convert stream into dataframe
    df = pd.read_csv(buffer)
    # clean up file 
    buffer.close()
    file.file.close()

    missing_columns = check_columns(df)
    if missing_columns:
      raise MissingColumnsException(missing_columns)
    else:
      logger.info("All required columns are present in the DataFrame")
      x_train, y_train = preprocess_train(dataset=df, target_width=int(model_time_frame.value))

      if model_time_frame.value == "365":
        model_path = 'models/year/365model.h5'
        logger.info("Using year ltv model...")
      elif model_time_frame.value == "90":
        model_path = 'models/quarter/90model.h5'
        logger.info("Using quarter ltv model...")
      else:
        model_path = 'models/month/30model.h5'
        logger.info("Using month ltv model...")

      # use keras to load in the correct file
      model = load_model(model_path, compile=False, custom_objects={'zero_inflated_lognormal_loss': zero_inflated_lognormal_loss})
      model.compile(loss=zero_inflated_lognormal_loss)
      
      # fit the model to the newly given data
      callback_patience = 5
      epochs = 100
      callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=callback_patience)
      ]

      model.fit(
        x_train,
        y_train,
        batch_size=1024,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
      )

      # write the model back to the path
      save_model(model=model, filepath=model_path)

      return JSONResponse(
        content={
          'status': 'Training job submitted successfully'
        }
      )
  except Exception as e:
    ERR_MSG = f'Exception encountered training model: {e}'
    logger.error(ERR_MSG)
    return JSONResponse(content={'status': ERR_MSG}, status_code=500)


