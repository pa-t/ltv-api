
class MissingColumnsException(Exception):
  def __init__(self, missing_columns):
    super().__init__(f"Missing columns: {', '.join(missing_columns)}")
    self.missing_columns = missing_columns
