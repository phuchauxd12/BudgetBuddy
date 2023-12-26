from datetime import date
from pydantic import BaseModel

class Bank(BaseModel):
    date: date
    total_income: float
    total_expenses: float