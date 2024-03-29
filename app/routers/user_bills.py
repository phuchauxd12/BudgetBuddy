from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from constant import USER_BILLS, Message
from models.user_bill import BillCreate, GetBillInformation
from bson import ObjectId
from datetime import datetime

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
    existing_bill = USER_BILLS.find_one({"user_id": ObjectId(user_id),'bill_name': bill_name})
    if existing_bill is not None:
        return JSONResponse(status_code=409, content={"message": "Bill already exists"})
    
    recurrent_date = datetime.strptime(recurrent_date_value, "%Y-%m-%d").date()
    if recurrent_date < datetime.now().date():
        return JSONResponse(status_code=400, content="Recurrent date cannot be in the past")
    
    USER_BILLS.insert_one(BillCreate(user_id= ObjectId(user_id), bill_name= bill_name, bill_value= bill_value, recurrent_reminder= recurrent_reminder,
                                 recurrent_date_value= recurrent_date_value).dict())
    return  JSONResponse(content= {"message": "Create bill successfully"})
 

@router.get("/user_bill/{user_id}", responses = {409: {"model": Message},
                                      422: {"model": Message},
                                      404: {"model": Message}})
def get_bill(user_id: str,):
    list_bills = USER_BILLS.find({"user_id": ObjectId(user_id)})
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
        return JSONResponse(status_code=404, content={'message': "User does not exist or recurrent remider is false."})   
    return JSONResponse(content=array)


@router.get("/user_bill/{user_id}/sorted", responses = {409: {"model": Message},
                                      422: {"model": Message},
                                      404: {"model": Message}})
def sorted_bill(user_id: str,):
    list_bills = USER_BILLS.find({"user_id": ObjectId(user_id)})
    array = [
            GetBillInformation(
                bill_id = str(value["_id"]),
                bill_name=value["bill_name"],
                bill_value=value["bill_value"],
                recurrent_date_value=value["recurrent_date_value"],
                recurrent_reminder=value["recurrent_reminder"]
            ).dict()
            for value in list_bills if value["recurrent_reminder"] == True 
        ]

    if not array:
        return JSONResponse(status_code=404, content={'message': "User does not exist or recurrent remider is false."})
    # sorted in ascending order of dates
    sorted_array = sorted(array, key=lambda x: datetime.strptime(x['recurrent_date_value'], '%Y-%m-%d'))    
    return JSONResponse(content=sorted_array)

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
        recurrent_date = datetime.strptime(recurrent_date_value, "%Y-%m-%d").date()
        if recurrent_date < datetime.now().date():
            return JSONResponse(status_code=400, content="Recurrent date cannot be in the past")
        update_data["recurrent_date_value"] = recurrent_date_value

    USER_BILLS.update_one({"user_id": ObjectId(user_id), "_id": ObjectId(bill_id)}, {"$set": update_data})
    return JSONResponse(status_code=200, content={"message": "Bill updated successfully"})

@router.delete("/user_bill/{user_id}/{bill_id}/delete")
def delete_bill(user_id: str,
    bill_id: str ,):
    
    USER_BILLS.delete_one({"user_id": ObjectId(user_id), "_id": ObjectId(bill_id)})
    return JSONResponse(status_code=200, content={"message": "Bill deleted successfully"}
        )
  