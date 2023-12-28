import datetime
from pydantic import BaseModel
from fastapi_users import schemas

class Bank(BaseModel):
    date: datetime.datetime
    total_income: float
    total_expenses: float