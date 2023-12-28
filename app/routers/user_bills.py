from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from constant import Constant, Message
from pymongo import MongoClient
from models.user_bill import BillCreate, GetBillInformation
from bson import ObjectId
from datetime import datetime
client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
db = client.get_collection("USER_BILLS")
router = APIRouter(prefix="/api/v1", tags=["CRUD User Bills"])

@router.post("/user_bill/{user_id}/create", responses ={409: {"model": Message},
                                            422: {"model": Message}})
def create_bill(
    user_id: str,
    bill_name: str = Form(..., description="Name of the bill"),
    bill_value: float = Form(..., description="Amount of the bill"),
    recurrent_reminder: bool = Form(..., description="Want to reminder about the bill"),
    recurrent_date_value: str = Form(..., description="Date of the bill"),
):
    existing_bill = db.find_one({"user_id": ObjectId(user_id),'bill_name': bill_name})
    if existing_bill is not None:
        return JSONResponse(status_code=409, content={"message": "Bill already exists"})
    
    try:
        datetime.strptime(recurrent_date_value, "%Y-%m-%d")
    except ValueError:
        return JSONResponse(status_code=422, content="Invalid date format. Please use YYYY-MM-DD.")

    try:
        db.insert_one(BillCreate(user_id= ObjectId(user_id), bill_name= bill_name, bill_value= bill_value, recurrent_reminder= recurrent_reminder,
                                 recurrent_date_value= recurrent_date_value).dict())
        return  JSONResponse(content= {"message": "Create bill successfully"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": f"Error creating bill: {str(e)}"})

@router.get("/user_bill/{user_id}", responses = {409: {"model": Message},
                                      422: {"model": Message},
                                      404: {"model": Message}})
def get_bill(user_id: str,):
    list_bills = db.find({"user_id": ObjectId(user_id)})
    array = [
            GetBillInformation(
                bill_id = str(value["_id"]),
                bill_name=value["bill_name"],
                bill_value=value["bill_value"],
                recurrent_date_value=value["recurrent_date_value"],
                recurrent_reminder=value["recurrent_reminder"]
            ).dict()
            for value in list_bills
        ]

    if not array:
        return JSONResponse(status_code=404, content={'message': "User does not exist."})
        
    return JSONResponse(content=array)


@router.put("/user_bill/{user_id}/{bill_id}/update")
def update_bill(
    user_id: str,
    bill_id: str,
    bill_name: str = Form(None, description="Name of the bill"),
    bill_value: float = Form(None, description="Amount of the bill"),
    recurrent_reminder: bool = Form(None, description="Want to reminder about the bill"),
    recurrent_date_value: str = Form(None, description="Date of the bill "),
):
    update_data = {}

    if bill_name is not None:
        update_data["bill_name"] = bill_name

    if bill_value is not None:
        update_data["bill_value"] = bill_value

    if recurrent_reminder is not None:
        update_data["recurrent_reminder"] = recurrent_reminder

    if recurrent_date_value is not None:
        try:
            datetime.strptime(recurrent_date_value, "%Y-%m-%d")
        except ValueError:
            return JSONResponse(status_code=422, content="Invalid date format. Please use YYYY-MM-DD.")
        update_data["recurrent_date_value"] = recurrent_date_value

    db.update_one({"user_id": ObjectId(user_id), "_id": ObjectId(bill_id)}, {"$set": update_data})
    return JSONResponse(status_code=200, content={"message": "Bill updated successfully"})

@router.delete("/user_bill/{user_id}/{bill_id}/delete")
def delete_bill(user_id: str,
    bill_id: str ,):
    
    db.delete_one({"user_id": ObjectId(user_id), "_id": ObjectId(bill_id)})
    return JSONResponse(status_code=200, content={"message": "Bill deleted successfully"}
        )
  