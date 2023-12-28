from fastapi import APIRouter, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from models.users import UserInfo
from constant import Message, Constant
from pymongo import MongoClient
from bson import ObjectId 
import bcrypt
import uuid
from fastapi.encoders import jsonable_encoder


from models.email import EmailSchema, send_email  
router = APIRouter(prefix="/api/v1", tags=["User Profile"])
client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
db = client.get_collection("USERS")

@router.get('/profile/{id}', responses= {404: {"model": Message},
                                        422: {"model": Message},
                                        401:{'model': Message}})
def profile(id: str):
    """
    Return the user data information
    """
    if len(id) ==24:
        account = db.find_one({"_id": ObjectId(id)})
        if account:
            try:
                # Convert ObjectId to string
                account["_id"] = str(account["_id"])
                return account
            except ValueError as e:
                return JSONResponse(status_code=402, content={"message": str(e)})
        else:
            return JSONResponse(status_code=404, content={"message": "User not found"})
    else:
        return  JSONResponse(status_code=401, content={"message": "Unauthorized"})


@router.put('/profile/{id}/update', responses= {404: {"model": Message},
                                                422: {"model": Message}})
def update( background_tasks: BackgroundTasks,
            id:str,
            password: str = Form(..., description="Password of the user"),
            number: str = Form(None, description="Phone number of the user"),
            address: str = Form(None, description="Address of the user"),):
    """
    Send the update data, it can be password, number, address
    """
    if len(id) == 24:
        NewKey = uuid.uuid1().hex
        NewPassword = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        account = db.find_one({"_id": ObjectId(id)})
        if account:
            try:
                if number is None:
                    number = account["number"]
                if address is None:
                    address = account["address"]
                str_password = NewPassword.decode('utf-8')
                # Store the hashed password in the database
                newAccount = db.find_one_and_update(
                    {"_id": ObjectId(id)},
                    {"$set": {"password": str_password,
                            "key": NewKey,
                            "number": number,
                            'address': address}},
                    return_document=True  # Return the updated document
                )

                # Convert ObjectId to string before returning the JSON response
                newAccount["_id"] = str(account["_id"])

                # Email send to the customer about the key for login
                subject = "key for account login in Budget Buddy application"
                body = "KEY: " + str(NewKey)
                data = EmailSchema(to=newAccount["email"], subject=subject, body=body).dict()
                background_tasks.add_task(send_email, data["to"], data["subject"], data["body"])

                return JSONResponse(content=jsonable_encoder(newAccount))
            except ValueError as e:
                return JSONResponse(status_code=422, content={"message": str(e)})
        else:
            return JSONResponse(status_code=404, content={"message": "User not found"})
    else:
        return JSONResponse(status_code=401, content={"message": "Unauthorized"})
