import datetime
from pydantic import BaseModel

class Bank(BaseModel):
    date: datetime.datetime
    total_income: float
    total_expenses: float