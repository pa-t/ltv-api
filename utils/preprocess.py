import pandas as pd
import numpy as np

from typing import List

ACCOUNTS_TO_REMOVE = [0, 4, 6, 7, 9, 118483]

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
  # TODO: we do is_refund == False ^ and then is_refund == True????
  dataset.loc[
    (dataset['is_refund'] == True) &
    (dataset['order_total'] > 0),
    'order_total'
  ]  *= -1 #subtract refund totals
  # TODO: what is this timestamp?
  dataset = dataset.loc[dataset['time_stamp'] < pd.to_datetime('2022-12-31')]
  total_rev = dataset.groupby('customer_id')['order_total'].sum()
  pos_rev = total_rev[total_rev >= 0]
  dataset = dataset[dataset['customer_id'].isin(pos_rev.index)]

  return dataset


def calc_total_spent(dataset: pd.DataFrame) -> List[pd.Series]:
  target_widths = [30, 90, 365]
  total_spent = []
  # calculate the cutoff date for each customer
  for width in target_widths:
    cutoffs = dataset.groupby('customer_id')['time_stamp'].min() + pd.Timedelta(days=width)

    # filter transactions after cutoff date for each customer
    mask = (dataset['time_stamp'] <= dataset['customer_id'].map(cutoffs))
    filtered_dataset = dataset.loc[mask]

    # calculate the total amount spent by each customer
    ltv = filtered_dataset.groupby('customer_id')['order_total'].sum().rename('ltv').astype(float)
    total_spent.append(ltv)
  
  return total_spent


def drop_overspenders(
  dataset: pd.DataFrame,
  target_width: int,
  targets_total_spent: List[pd.Series]
) -> pd.DataFrame:
  """
    Filter out accounts taht spend over the theoretical limit,
    this is typically an indicator of test accounts
  """
  # select the cutoff amount
  if target_width == 365:
    total_spent = targets_total_spent[2]
  elif target_width == 90:
    total_spent = targets_total_spent[1]
  else:
    total_spent = targets_total_spent[0]
  
  total_cutoff = total_spent > target_width + 20
  total_overspent = total_spent[total_cutoff]
  df = dataset.drop(total_overspent.index)

  return df


def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
  """
    Series of preprocessing steps, takes raw data from db and prepares it for use
  """
  # convert timestamp column to be datetime object
  dataset['time_stamp'] = pd.to_datetime(dataset['time_stamp'])
  # filter out dirty data
  dataset = filter_data(dataset=dataset)
  # calc total spents
  targets_total_spent = calc_total_spent(dataset=dataset)

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
  version = customer_data['version'].first().astype(str)
  campaign_id = customer_data['campaign_id'].first().astype(float)
  domain = customer_data['email_address'].first().str.split('@').str[1].astype(str)
  first_on_hold = customer_data['on_hold'].first()

  df = pd.DataFrame({
    'afid': afid.astype(str),
    'cc_type': cc_type.astype(str),
    'main_product_id': main_product_id.astype(int),
    'campaign_id': campaign_id.astype(int),
    'first_order_amount': first_order_amount.astype(float),
    'domain': domain.astype(str),
    'total_spent_30': targets_total_spent[0].astype(float),
    'total_spent_90': targets_total_spent[1].astype(float),
    'total_spent_365': targets_total_spent[2].astype(float),
    'version': version.astype(str),
    'first_on_hold': first_on_hold.astype(str)
  })

  # TODO: where should this come from, the request?
  TARGET_WIDTH = 365

  # drop the overspending accounts
  df = drop_overspenders(
    dataset=df,
    target_width=TARGET_WIDTH,
    targets_total_spent=targets_total_spent
  )

  # Determine if email domain is gmail, yahoo, or other
  top_domains = domain.value_counts().nlargest(3).index
  df['domain'] = np.where(df['domain'].isin(top_domains),df['domain'], 'other')

  time_cutoff = dataset['time_stamp'].max() - pd.Timedelta(TARGET_WIDTH, 'D')
  customer_window = dataset.groupby('customer_id')['time_stamp'].first() > time_cutoff
  df.drop(df[customer_window].index)

  # Add affiliate id and their respective funnel domain (google, bing, email, etc)
  # TODO: figure out where to get this csv from
  # afid_df = pd.read_csv('./drive/MyDrive/afid_mapping_ath.csv')
  # df = pd.merge(df, afid_df, on='afid', how='left').set_index(df.index)
  # top_sources = df['source_system_description'].value_counts().nlargest(10).index
  # df['source_system_description'] = np.where(df['source_system_description'].isin(top_sources),df['source_system_description'], 'Other')

  # Restrict the afid column to the top 20 values or 'other'
  top_afid = df['afid'].value_counts().nlargest(20).index
  df['afid'] = np.where(df['afid'].isin(top_afid), df['afid'], 'other')

  # add in billing state column
  # TODO: figure out where to get this csv from
  # billing_state = pd.read_csv('./drive/MyDrive/ath_billing_state.csv')
  # billing_state = billing_state.set_index('customer_id')
  # billing_state = billing_state.groupby('customer_id').first()
  # df = df.merge(billing_state, on='customer_id')

  # fill na values
  df = df.fillna('Other')

  return df