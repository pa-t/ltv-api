from pydantic import BaseModel


class RecordSchema(BaseModel):
  # TODO: what if not all should be optional?
  customer_id: int # TODO: are they ints?
  afid: str
  cc_type: str
  main_product_id: int
  campaign_id: int
  first_order_amount: float
  domain: str
  total_spent_30: float
  total_spent_90: float
  total_spent_365: float
  version: str
  first_on_hold: str
  time_range: str # TODO: should this be passed here? should we have diff endpoints?


class PredictionSchema(BaseModel):
  # TODO: what does the response from the model look like / how can we get it in this format?
  customer_id: str # TODO: are they ints?
  predicted_ltv: float
  time_range: str # TODO: do we need this?