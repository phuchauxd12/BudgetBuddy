import datetime
from pydantic import BaseModel, Field


class Bank(BaseModel):
    date: datetime.datetime
    total_income: float =  Field(..., le=10000.0)
    total_expenses: float = Field(..., le=10000.0)
    
    
