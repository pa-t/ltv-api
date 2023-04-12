import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def zero_inflated_lognormal_pred(logits: tf.Tensor) -> tf.Tensor:
  """Calculates predicted mean of zero inflated lognormal logits.
  Arguments:
    logits: [batch_size, 3] tensor of logits.
  Returns:
    preds: [batch_size, 1] tensor of predicted mean.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = tf.keras.backend.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = tf.keras.backend.softplus(logits[..., 2:])
  preds = (
      positive_probs *
      tf.keras.backend.exp(loc + 0.5 * tf.keras.backend.square(scale)))
  return preds


def zero_inflated_lognormal_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
  """Computes the zero inflated lognormal loss.
  Usage with tf.keras API:
  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=zero_inflated_lognormal)
  ```
  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].
  Returns:
    Zero inflated lognormal loss value.
  """
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  logits.shape.assert_is_compatible_with(
      tf.TensorShape(labels.shape[:-1].as_list() + [3]))

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.losses.binary_crossentropy(
      y_true=positive, y_pred=positive_logits, from_logits=True)

  loc = logits[..., 1:2]
  scale = tf.math.maximum(
    tf.keras.backend.softplus(logits[..., 2:]),
    tf.math.sqrt(tf.keras.backend.epsilon()))
  safe_labels = positive * labels + (
      1 - positive) * tf.keras.backend.ones_like(labels)
  regression_loss = -tf.keras.backend.mean(
      positive * tfd.LogNormal(loc=loc, scale=scale).log_prob(safe_labels),
      axis=-1)

  return classification_loss + regression_loss


def feature_dict(df, numerical_features, categorical_features):
    """
    Converting dataFrame to dictionary for model inputs
    """
    features = {k: v.values for k, v in dict(df[categorical_features]).items()}
    features["numeric"] = df[numerical_features].values
    return features


def model_predict(model, data, feature_map):
    """
    Function to make predictions on out of sample data. You have to encode the categorical data the same
    way as your training set. This function lets you do that along with optionally calculating
    performance metrics for on your new data.

    `model`: trained model to be used for prediction
    `data`: either pandas DataFrame or a link to a csv or parquet file
    `feature_map`: The feature mapping variable you got when running the preprocess() function when creating 
                train-test split. This is required to create identical encodings on the new data
    `show_performance`: Default False. Whether to calculate performance metrics on the new data
    """
    ##Reading in data if not a pandas DataFrame
    if isinstance(data, str):
        path, file_type = os.path.splitext(data)
        if file_type==".csv":
            data=pd.read_csv(data)
        elif file_type==".parquet":
            data=pd.read_parquet(data, engine='pyarrow')

    all_variables = feature_map["categorical_features"] + feature_map["numerical_features"] + feature_map["day1_purchaseAmt_col"]

    for col in all_variables:
        if col not in data.columns:
            raise ValueError(f"Error: {col} column not found in `data`. Please keep all column names identical to the one used while modeling")
 
    data = data[all_variables]

    for cat in feature_map["categorical_features"]:
        levels=list(feature_map[cat].keys())
        ##Replacing new categorical levels with UNDEFINED
        data[cat] = data[cat].apply( lambda t: t if t in levels else 'UNDEFINED')
        # Mappings levels to the corresponding number.
        data[cat] = data[cat].apply( lambda t: feature_map[cat][t])

    x_test = feature_dict(data, feature_map["numerical_features"], feature_map["categorical_features"])
    x_test = { feat: np.array(x_test[feat]) for feat in x_test.keys()}

    logits = model.predict(x_test, batch_size=1024)

    ltv_pred = zero_inflated_lognormal_pred(logits).numpy().flatten()

    return pd.DataFrame({
        'ltv_prediction': ltv_pred
    })
