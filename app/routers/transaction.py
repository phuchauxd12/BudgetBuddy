from pymongo import MongoClient
import pandas as pd
import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from models.transaction import Transaction, GetTransactionInformation
from bson import ObjectId 
from constant import Constant

router = APIRouter(prefix="/api/v1", tags=["Import Transaction"])
client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
db = client.get_collection("TRANSACTION_HISTORY")
SAVING_COLLECTION = client.get_collection("SAVING_ACCOUNTS")
INVESTMENT_COLLECTION = client.get_collection("INVESTMENT_ACCOUNTS")
EXPENSE_COLLECTION = client.get_collection("EXPENSE_ACCOUNTS")
PLAN_EXPENSE_COLLECTION = client.get_collection("EXPENSE_SPENDING")
GOAL_SETTINGS = client.get_collection("GOAL_SETTINGS")

MODEL = {
    "Saving": SAVING_COLLECTION,
    "Investment": INVESTMENT_COLLECTION,
    "Expense": EXPENSE_COLLECTION,
}

def update_expense_value_transaction(transaction_type, amount, user_id):
    expense = PLAN_EXPENSE_COLLECTION.find_one({"user_id": ObjectId(user_id), "category": transaction_type})
    if expense is not None:
        PLAN_EXPENSE_COLLECTION.update_one(
            {"user_id": ObjectId(user_id), "category": transaction_type},
            {"$set": {
                "current_total_use": expense["current_total_use"] + amount
            }}
        )

def update_goal_settings(user_id, amount):
    goal = GOAL_SETTINGS.find_one({"user_id": ObjectId(user_id)})
    if goal is not None:
        GOAL_SETTINGS.update_one(
            {"user_id": ObjectId(user_id)},
            {"$set": {
                "saving_amount": goal["saving_amount"] + amount
            }}
        )

@router.post('/transaction/{user_id}/create')
def post_transaction(user_id: str,
                    transaction_name: str = Form(..., description="Name of the transaction"),
                    Payee: str = Form(..., description="Payee of the transaction"),
                    transaction_date: str = Form(..., description="Date of the transaction"),
                    Amount: float = Form(..., description="Amount of the transaction"),
                    account_name: str = Form(..., description="Name of the account"),
                    account_type: str = Form(..., description="Type of the account"),
                    transaction_type: str = Form(..., description="Type of the transaction"),):

    if account_type not in MODEL:
        raise HTTPException(status_code=422, detail="Invalid account type")
    
    # Check if account exists
    account = MODEL[account_type].find_one({"user_id": ObjectId(user_id), "account_name": account_name})
    if account is None:
        raise HTTPException(status_code=404, detail="Account does not exist.")
    account_id = account["_id"]
            
    # Check if the transaction already exists 
    existing_transaction = db.find_one({
        "transaction_name": transaction_name,
        "Payee": Payee,
        "transaction_date": transaction_date,
        "Amount": Amount,
        "account_name": account_name,
        "account_type": account_type,
        "account_id": ObjectId(account_id),
        "transaction_type": transaction_type,
        "user_id": ObjectId(user_id)
    })
    if existing_transaction is not None:
        raise HTTPException(status_code=409, detail="Transaction already exists")
    
    db.insert_one(
        Transaction(transaction_name= transaction_name, Payee= Payee, transaction_date= transaction_date, Amount= Amount,
                    account_name= account_name, account_type= account_type,account_id=ObjectId(account_id), 
                    transaction_type= transaction_type, user_id= ObjectId(user_id)
                    ).dict()
    )
    
    if transaction_type == "Income":
        MODEL[account_type].update_one(
            {"user_id": ObjectId(user_id), "account_name": account_name},
            {"$inc": {"current_balance": Amount}}
        )
    else:
        MODEL[account_type].update_one(
            {"user_id": ObjectId(user_id), "account_name": account_name},
            {"$inc": {"current_balance": -Amount}}
        )
        update_expense_value_transaction(transaction_type, Amount, user_id)
    return JSONResponse(content={"message": "Transaction created successfully"})


@router.post('/transaction/{user_id}/import_csv')
def import_transaction(user_id: str,csv_file: UploadFile = File(...)):
    try:
        # Read the CSV file
        headers = ['transaction_name','Payee','transaction_date','Amount','account_name','account_type','transaction_type']
        df = pd.read_csv(BytesIO(csv_file.file.read()), names=headers, skiprows= 1)
        data = df.to_dict(orient='records')

        for value in data:

            if value["account_type"] not in MODEL:
                return JSONResponse(status_code=422, content={"message": "Invalid account type"})

            # Check if account exist
            account = MODEL[value["account_type"]].find_one({"user_id": ObjectId(user_id), "account_name": value["account_name"]})
            if account is None:
                return JSONResponse(status_code=404, content={'message': "Account does not exist."})
            account_id = account["_id"]
            
            # Check if the transaction already exist 
            existing_transaction = db.find_one({
                "transaction_name": value['transaction_name'],
                "Payee": value['Payee'],
               "transaction_date":value["transaction_date"],
                "Amount": value["Amount"],
                "account_name": value["account_name"],
                "account_type": value["account_type"],
                "account_id": ObjectId(account_id),
                "transaction_type": value["transaction_type"],
                "user_id": ObjectId(user_id)
            })
            if existing_transaction is not None:
                return JSONResponse(status_code=409, content={"message": "Transaction already exists"})
            
            db.insert_one(
                Transaction(transaction_name= value['transaction_name'],Payee= value['Payee'], transaction_date= value["transaction_date"], Amount= value["Amount"],
                            account_name= value["account_name"],account_type= value["account_type"],account_id=ObjectId(account_id), 
                            transaction_type= value["transaction_type"], user_id= ObjectId(user_id)
                            ).dict()
            )
        return JSONResponse(content={"message": "Transaction created successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@router.get("/transaction/{user_id}")
def get_transaction(user_id: str):
    list_transaction = db.find({"user_id": ObjectId(user_id)})
    array = [
           GetTransactionInformation(
                transaction_id = str(value["_id"]),
                transaction_name=value["transaction_name"],
                transaction_date=value["transaction_date"],
                Payee=value["Payee"],
                Amount=value["Amount"],
                account_name=value["account_name"],
                account_type=value["account_type"],
                transaction_type=value["transaction_type"],
            ).dict()
            for value in list_transaction
        ]

    if not array:
        return JSONResponse(status_code=404, content={'message': "Transaction does not exist."})
        
    return JSONResponse(content=array)


@router.put("/transaction/{user_id}/{transaction_id}/update")
def update_transaction(
    user_id: str,
    transaction_id: str,
    transaction_name: str = Form(None, description="Name of the transaction"),
    Payee: str = Form(None, description="Payee of the transaction"),
    transaction_date: str = Form(None, description="Date of the transaction"),
    Amount: float = Form(None, description="Amount of the transaction"),
    account_name: str = Form(None, description="Name of the account"),
    account_type: str = Form(None, description="Type of the account"),
    transaction_type: str = Form(None, description="Type of the transaction"),
):
    try:
        transaction = db.find_one({"_id": ObjectId(transaction_id)})
        update_data = {}

         # Check if user changes the account_type and account name
        if account_type is not None and account_name is not None:
            if account_type not in MODEL:
                return JSONResponse(status_code=422, content={"message": "Invalid account type"})
            account = MODEL[account_type].find_one({"user_id": ObjectId(user_id), "account_name": account_name})
            if account is None:
                return JSONResponse(status_code=404, content={'message': "Account does not exist."})
            update_data["account_type"] = account_type
            update_data["account_name"] = account_name
            update_data["account_id"] = account["_id"]
            
        # Check if user only s the account_type
        elif account_type is not None:
            if account_type not in MODEL:
                return JSONResponse(status_code=422, content={"message": "Invalid account type"})
            account = MODEL[account_type].find_one({"user_id": ObjectId(user_id), "account_name": transaction["account_name"]})
            if account is None:
                return JSONResponse(status_code=404, content={'message': "Account does not exist."})
            update_data["account_type"] = account_type
            update_data["account_id"] = account["_id"]
                 
        # Check if user only changes the account_type
        elif account_name is not None:
            account = MODEL[transaction["account_type"]].find_one({"user_id": ObjectId(user_id), "account_name": account_name})
            if account is None:
                return JSONResponse(status_code=404, content={'message': "Account does not exist."})
            update_data["account_name"] = account_name
            update_data["account_id"] = account["_id"]

        if transaction_name is not None:
            update_data["transaction_name"] = transaction_name

        if Payee is not None:
            update_data["Payee"] = Payee

        if Amount is not None:
            update_data["Amount"] = Amount

        if transaction_type is not None:
            update_data["transaction_type"] = transaction_type

        if transaction_date is not None:
            update_data["transaction_date"] = transaction_date

        # Update the transaction
        db.update_one(
                {"_id": ObjectId(transaction_id), "user_id": ObjectId(user_id)},
                {"$set": update_data}
            )
        return JSONResponse(content={"message": "Transaction updated successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})



@router.delete("/transaction/{user_id}/{transaction_id}/delete")
def delete_transaction(user_id: str, transaction_id: str):
    try:
        # Delete the transaction
        db.find_one_and_delete({"_id": ObjectId(transaction_id), "user_id": ObjectId(user_id)})

        return JSONResponse(content={"message": "Transaction deleted successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

