from __future__ import annotations

from pydantic import BaseModel, Field
from bson import ObjectId
from models.goal import GoalModelView
from typing import Optional

class AccountGeneric(BaseModel):
    user_id: ObjectId = Field(..., description="User id of the bank account")
    bank_id: ObjectId = Field(..., description="Bank id of the bank account")
    number_of_account: int = Field(1, description="Number of account of the bank account")
    #total_balance: float = Field(0, description="Total balance of the bank account")
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed=True
        json_encoders = {
            ObjectId: str
        }

class BaseAccount(BaseModel):
    account_id : ObjectId = Field(..., description="Account id of the bank account")
    account_name: str = Field(..., description="Account name of the bank account")
    user_id: ObjectId = Field(..., description="User id of the bank account")
    bank_id: ObjectId = Field(..., description="Bank id of the bank account")
    current_balance: float = Field(..., description="Current balance of the bank account")
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed=True
        json_encoders = {
            ObjectId: str
        }

class SavingOrInvestmentAccount(BaseAccount):
    goal_id: Optional[str] = Field(None, description="Goal id of the bank account")

class ExpenseAccount(BaseAccount):
    tag_id: str = Field(None, description="Tag id of the bank account")
    
class GetAccountInformation(BaseModel):
    account_name: str = Field(..., description="Account name of the bank account")
    bank_name: str = Field(..., description="Bank name of the bank account")
    current_balance: float = Field(..., description="Current balance of the bank account")
    goal_information: Optional[GoalModelView]
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed=True
        json_encoders = {
            ObjectId: str
        }

class GetAllAccountName(BaseModel):
    list_account_name: list = Field(..., description="List of account name of the bank account")