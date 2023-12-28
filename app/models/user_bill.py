from pydantic import BaseModel, Field, validator
from bson import ObjectId
from datetime import datetime 

class BillCreate(BaseModel):
    user_id: ObjectId = Field(..., description= "ID of the user")
    bill_name: str = Field(...,description="Name of the bill")
    bill_value: float = Field(...,description="Amount of the bill")
    recurrent_reminder: bool= Field(default= False, description="Want to reminder about the bill")
    recurrent_date_value: str = Field(..., description="Date of the bill")


    
    
    class Config:
            allow_population_by_field_name = True
            arbitrary_types_allowed=True
            json_encoders = {
                ObjectId: str
            }

class GetBillInformation(BaseModel):
    bill_id: str = Field(alias="bill_id", default=None)
    bill_name: str = Field(..., description="Bill name of user")
    bill_value: float = Field(..., description="Amount bill of user")
    recurrent_reminder: bool= Field(default= False, description="Want to reminder about the bill")
    recurrent_date_value: str = Field(..., description="Date of the bill")
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed=True
        json_encoders = {
            ObjectId: str
        }