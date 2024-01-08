import pandas as pd
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from models.transaction import Transaction, GetTransactionInformation
from bson import ObjectId 
from constant import TRANSACTION_COLLECTION, SAVING_COLLECTION, INVESTMENT_COLLECTION, EXPENSE_COLLECTION, EXPENSE_SPENDING_COLLECTION, GOAL_SETTINGS
from datetime import datetime
router = APIRouter(prefix="/api/v1", tags=["Import Transaction"])

MODEL = {
    "Saving": SAVING_COLLECTION,
    "Investment": INVESTMENT_COLLECTION,
    "Expense": EXPENSE_COLLECTION,
}

def update_expense_value_transaction(category, amount, user_id, time_duration):
    expense = EXPENSE_SPENDING_COLLECTION.find_one({"user_id": ObjectId(user_id), "category": category,"time_duration":time_duration})
    if expense is not None:
        EXPENSE_SPENDING_COLLECTION.update_one(
            {"user_id": ObjectId(user_id), "category": category, "time_duration": time_duration},
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
                    transaction_type: str = Form(..., description="Type of the transaction"),
                    category: str = Form(None, description="category of the transaction"),):

    if account_type not in MODEL:
        raise HTTPException(status_code=422, detail="Invalid account type")
    
    # Check if account exists
    account = MODEL[account_type].find_one({"user_id": ObjectId(user_id), "account_name": account_name})
    if account is None:
        raise HTTPException(status_code=404, detail="Account does not exist.")
    account_id = account["_id"]
            
    # Check if the transaction already exists 
    existing_transaction = TRANSACTION_COLLECTION.find_one({
        "transaction_name": transaction_name,
        "Payee": Payee,
        "transaction_date": transaction_date,
        "Amount": Amount,
        "account_name": account_name,
        "account_type": account_type,
        "account_id": ObjectId(account_id),
        "transaction_type": transaction_type,
        "user_id": ObjectId(user_id),
        "category": category,
    })
    if existing_transaction is not None:
        raise HTTPException(status_code=409, detail="Transaction already exists")
    
    TRANSACTION_COLLECTION.insert_one(
        Transaction(transaction_name= transaction_name, Payee= Payee, transaction_date= transaction_date, Amount= Amount,
                    account_name= account_name, account_type= account_type,account_id=ObjectId(account_id), 
                    transaction_type= transaction_type, user_id= ObjectId(user_id), category= category
                    ).dict()
    )
    
    if transaction_type in ["Income","Investment"]:
        MODEL[account_type].update_one(
            {"user_id": ObjectId(user_id), "account_name": account_name},
            {"$inc": {"current_balance": Amount}}
        )
    if transaction_type == "Outcome":
        MODEL[account_type].update_one(
            {"user_id": ObjectId(user_id), "account_name": account_name},
            {"$inc": {"current_balance": -Amount}}
        )
        parsed_date = datetime.strptime(transaction_date, "%Y-%m-%d")
        formatted_date = parsed_date.strftime("%b %Y")
        update_expense_value_transaction(category, Amount, user_id,formatted_date)
    return JSONResponse(content={"message": "Transaction created successfully"})


@router.post('/transaction/{user_id}/import_csv')
def import_transaction(user_id: str,csv_file: UploadFile = File(...)):
    try:
        # Read the CSV file
        headers = ['transaction_name','Payee','transaction_date','Amount','account_name','account_type','transaction_type', "category"]
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
            existing_transaction = TRANSACTION_COLLECTION.find_one({
                "transaction_name": value['transaction_name'],
                "Payee": value['Payee'],
               "transaction_date":value["transaction_date"],
                "Amount": value["Amount"],
                "account_name": value["account_name"],
                "account_type": value["account_type"],
                "account_id": ObjectId(account_id),
                "transaction_type": value["transaction_type"],
                "user_id": ObjectId(user_id),
                "category": value["category"],
            })
            if existing_transaction is not None:
                return JSONResponse(status_code=409, content={"message": "Transaction already exists"})
            
            TRANSACTION_COLLECTION.insert_one(
                Transaction(transaction_name= value['transaction_name'],Payee= value['Payee'], transaction_date= value["transaction_date"], Amount= value["Amount"],
                            account_name= value["account_name"],account_type= value["account_type"],account_id=ObjectId(account_id), 
                            transaction_type= value["transaction_type"], user_id= ObjectId(user_id), category= value["category"]
                            ).dict()
            )
            if value["transaction_type"] == "Income":
                MODEL[value["account_type"]].update_one(
                    {"user_id": ObjectId(user_id), "account_name": value["account_name"]},
                    {"$inc": {"current_balance": value["Amount"]}}
                )
            else:
                MODEL[value["account_type"]].update_one(
                    {"user_id": ObjectId(user_id), "account_name": value["account_name"]},
                    {"$inc": {"current_balance": -value["Amount"]}}
                )
                parsed_date = datetime.strptime(value["transaction_date"], "%Y-%m-%d")
                formatted_date = parsed_date.strftime("%b %Y")
                update_expense_value_transaction(value["category"], value["Amount"], user_id, formatted_date)

        return JSONResponse(content={"message": "Transaction created successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@router.get("/transaction/{user_id}")
def get_transaction(user_id: str):
    list_transaction = TRANSACTION_COLLECTION.find({"user_id": ObjectId(user_id)})
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
                category= value["category"],
            ).dict()
            for value in list_transaction
        ]

    if not array:
        return JSONResponse(status_code=404, content={'message': "Transaction does not exist."})
        
    return JSONResponse(content=array)


@router.get("/transaction/{user_id}/{year_month}")
def get_transaction_by_month(user_id: str,
                    year_month: str ,
                    ):
    
    list_transaction = TRANSACTION_COLLECTION.find({"user_id": ObjectId(user_id)})
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
                category= value["category"],
            ).dict()
            for value in list_transaction if year_month == '-'.join(value["transaction_date"].split('-')[:2])
        ]
    if not array:
        return JSONResponse(status_code=404, content={'message': "Transaction does not exist."})
        
    return JSONResponse(content=array)
@router.delete("/transaction/{user_id}/{transaction_id}/delete")
def delete_transaction(user_id: str, transaction_id: str):
    try:
        transaction = TRANSACTION_COLLECTION.find_one({"_id": ObjectId(transaction_id), "user_id": ObjectId(user_id)})
        Amount = -transaction["Amount"]
        if transaction["transaction_type"] == "Income":
            MODEL[transaction["account_type"]].update_one(
                {"user_id": ObjectId(user_id), "account_name": transaction["account_name"]},
                {"$inc": {"current_balance": Amount}}
            )
        else:
            MODEL[transaction["account_type"]].update_one(
                {"user_id": ObjectId(user_id), "account_name": transaction["account_name"]},
                {"$inc": {"current_balance": -Amount}}
            )
        parsed_date = datetime.strptime(transaction["transaction_date"], "%Y-%m-%d")
        formatted_date = parsed_date.strftime("%b %Y")
        update_expense_value_transaction(transaction["category"], Amount, user_id, formatted_date)
        # Delete the transaction
        TRANSACTION_COLLECTION.find_one_and_delete({"_id": ObjectId(transaction_id), "user_id": ObjectId(user_id)})

        return JSONResponse(content={"message": "Transaction deleted successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
