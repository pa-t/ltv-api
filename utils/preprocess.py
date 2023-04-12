import pandas as pd
from utils.zltv_model import model_predict, dnn_model, embedding_layer, embedding_dim, feature_dict
from utils.zltv_model import preprocess as zltv_preprocess
from typing import List
import pickle


ACCOUNTS_TO_REMOVE = [0, 4, 6, 7, 9, 118483]
CATEGORICAL_FEATURES = ['afid', 'cc_type', 'main_product_id','campaign_id', 'billing_state']
NUMERIC_FEATURES = ['first_order_amount']
TARGET='total_spent'
DAY1_PURCHASEAMT_COL='first_order_amount'

def filter_data(dataset: pd.DataFrame) -> pd.DataFrame:
  """
    Performs some filtering on the dataset to remove records that could jeopardize training
  """
  #remove test accounts and outlier
  dataset = dataset[~dataset['customer_id'].isin(ACCOUNTS_TO_REMOVE)]
  # remove any fraud, chargeback or refund transactions
  dataset = dataset.loc[
    (dataset['is_fraud'] == False) &
    (dataset['is_chargeback'] == False) &
    (dataset['is_refund'] == False)
  ]
  total_rev = dataset.groupby('customer_id')['order_total'].sum()
  pos_rev = total_rev[total_rev >= 0]
  dataset = dataset[dataset['customer_id'].isin(pos_rev.index)]

  return dataset


def calc_total_spent(dataset: pd.DataFrame, target_width = None) -> List[pd.Series]:
  if target_width is None:
    target_widths = [30, 90, 365]
  else:
    target_widths = [target_width]
  total_spent = []
  # calculate the cutoff date for each customer
  for width in target_widths:
    cutoffs = dataset.groupby('customer_id')['time_stamp'].min() + pd.Timedelta(days=width)

    # filter transactions after cutoff date for each customer
    mask = (dataset['time_stamp'] <= dataset['customer_id'].map(cutoffs))
    filtered_dataset = dataset.loc[mask]

    # calculate the total amount spent by each customer
    ltv = filtered_dataset.groupby('customer_id')['order_total'].sum().rename(f'total_spent_{width}').astype(float)
    total_spent.append(ltv)
  
  return total_spent


def drop_overspenders(
  dataset: pd.DataFrame,
  target_width: int,
  targets_total_spent: List[pd.Series]
) -> pd.DataFrame:
  """
    Filter out accounts that spend over the theoretical limit,
    this is typically an indicator of test accounts
  """
  threshold = 20
  # select the cutoff amount
  if target_width == 365:
    total_spent = targets_total_spent[2]
  elif target_width == 90:
    total_spent = targets_total_spent[1]
  else:
    total_spent = targets_total_spent[0]
  
  total_cutoff = total_spent > target_width + threshold
  total_overspent = total_spent[total_cutoff]
  df = dataset.drop(total_overspent.index)

  return df


def preprocess_train(dataset: pd.DataFrame, target_width: int) -> pd.DataFrame:
  """
    Series of preprocessing steps, takes raw data from db and prepares it for use
  """
  df = get_features(dataset=dataset, target_width=target_width)
  targets_total_spent = calc_total_spent(dataset=dataset, target_width=target_width)
  # drop the overspending accounts
  df = drop_overspenders(
    dataset=df,
    target_width=target_width,
    targets_total_spent=targets_total_spent
  )
  df = df.join(targets_total_spent)

  with open('feature_map.pkl', 'rb') as f:
    feature_map = pickle.load(f)

  all_variables= feature_map["categorical_features"]+feature_map["numerical_features"]+[feature_map["target"] , feature_map["day1_purchaseAmt_col"]]
  
  for col in all_variables:
      if col not in df.columns:
          raise ValueError("Error -"+ col +" column not found in `df`. Please keep all column names identical to the one used while modelling ")
  
  
  df=df[all_variables]
  if df[feature_map["target"]].dtype!="float32":
      df[feature_map["target"]]=df[feature_map["target"]].astype("float32")
  
  for cat in feature_map["categorical_features"]:
      levels=list(feature_map[cat].keys())
      ##Replacing new categorical levels with UNDEFINED
      df[cat] = df[cat].apply( lambda t: t if t in levels else 'UNDEFINED' )
      # Mappings levels to the corresponding number.
      df[cat] = df[cat].apply( lambda t: feature_map[cat][t]) 

  x_train=feature_dict(df, feature_map["numerical_features"], feature_map["categorical_features"])
  x_train = { feat: np.array(x_train[feat]) for feat in x_train.keys()}
  return x_train, targets_total_spent


def get_features(dataset: pd.DataFrame, target_width: int) -> pd.DataFrame:
  dataset['time_stamp'] = pd.to_datetime(dataset['time_stamp'])
  # filter out dirty data
  dataset = filter_data(dataset=dataset)

  # groupby customer id
  customer_data = dataset.groupby('customer_id')

  # create series for final dataset
  first_order_amount = dataset.loc[
    dataset.groupby('customer_id')['time_stamp'].idxmin()
  ][['customer_id', 'order_total']]
  first_order_amount = first_order_amount.set_index(
    'customer_id', drop = True
  ).squeeze().rename('first_order_amount')
  afid = customer_data['afid'].first().astype(str)
  cc_type = customer_data['cc_type'].first().astype(str)
  main_product_id = customer_data['main_product_id'].first().astype(float)
  campaign_id = customer_data['campaign_id'].first().astype(float)
  billing_state = customer_data['billing_state'].first().astype(str)
 
  df = pd.DataFrame({
    'afid': afid.astype(str),
    'cc_type': cc_type.astype(str),
    'main_product_id': main_product_id.astype(int),
    'campaign_id': campaign_id.astype(int),
    'first_order_amount': first_order_amount.astype(float),
    'billing_state': billing_state.first().astype(str)
  })

  time_cutoff = dataset['time_stamp'].max() - pd.Timedelta(target_width, 'D')
  customer_window = dataset.groupby('customer_id')['time_stamp'].first() > time_cutoff
  df.drop(df[customer_window].index)

  # fill na values
  df = df.fillna('Other')
  return df

def check_columns(df: pd.DataFrame):
  required_columns = [
    'customer_id', 'afid', 'campaign_id', 'cc_type', 'on_hold',
    'order_total', 'main_product_id', 'billing_state'
  ]

  missing_columns = [column for column in required_columns if column not in df.columns]

  if missing_columns:
      return True
  return False