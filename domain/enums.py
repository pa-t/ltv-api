from enum import Enum

class ModelTimeFrame(str, Enum):
  year = "365"
  quarter = "90"
  month = "30"