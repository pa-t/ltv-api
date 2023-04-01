import pandas as pd

ACCOUNTS_TO_REMOVE = [0, 4, 6, 7, 9, 118483]

def filter_data(dataset: pd.DataFrame) -> pd.DataFrame:
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


