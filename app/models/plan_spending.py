from __future__ import annotations
from pydantic import BaseModel, Field
from bson import ObjectId

class PlannedSpendingModel(BaseModel):
    spending_name: str = Field(..., description = "Name of the spending")
    amount: int | float = Field(..., description = "Amount of the spending")
    user_id: ObjectId = Field(..., description = "User id of the spending")
    spending_type: str = Field(..., 
                               description = "Type of the spending. This should either be Income, Bills, Subscription or Credit Card Payment")
    time_duration: str = Field(..., description = "Time duration of the spending planning. Example: Jan 2024, Feb 2024, etc.")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class PlannedSpendingModelView(BaseModel):
    spending_name: str = Field(..., description = "Name of the spending")
    amount: int | float = Field(..., description = "Amount of the spending")
    spending_type: str = Field(..., 
                               description = "Type of the spending. This should either be Income, Bills, Subscription or Credit Card Payment")
    time_duration: str = Field(..., description = "Time duration of the spending planning. Example: Jan 2024, Feb 2024, etc.")

class MonthlyExpensePlan(BaseModel):
    user_id: ObjectId = Field(..., description = "User id of the spending")
    category: str = Field(..., description = "Category of the spending")
    initial_amount: int | float = Field(..., description = "Initial amount of the spending")
    current_total_use: int | float = Field(..., description = "Current total use of the spending")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MonthlyExpensePlanModelView(BaseModel):
    category: str = Field(..., description = "Category of the spending")
    initial_amount: int | float = Field(..., description = "Initial amount of the spending")
    current_total_use: int | float = Field(..., description = "Current total use of the spending")