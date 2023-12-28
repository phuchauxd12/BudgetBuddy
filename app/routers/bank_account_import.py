from __future__ import annotations

from fastapi import APIRouter, Form, File, UploadFile
import pandas as pd
from io import BytesIO
from fastapi.responses import JSONResponse
from constant import Constant
from pymongo import MongoClient
from models.bank_account import (
    SavingOrInvestmentAccount,
    ExpenseAccount,
    AccountGeneric,
    BaseAccount,
    GetAccountInformation,
)
from bson import ObjectId

# MONGODB CONNECTION
CLIENT = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
BANK_COLLECTION = CLIENT.get_collection("BANK_INFO")
ACCOUNT_COLLECTION = CLIENT.get_collection("ACCOUNTS")
SAVING_COLLECTION = CLIENT.get_collection("SAVING_ACCOUNTS")
INVESTMENT_COLLECTION = CLIENT.get_collection("INVESTMENT_ACCOUNTS")
EXPENSE_COLLECTION = CLIENT.get_collection("EXPENSE_ACCOUNTS")

MODEL = {
    "Saving": [SavingOrInvestmentAccount, SAVING_COLLECTION],
    "Investment": [SavingOrInvestmentAccount, INVESTMENT_COLLECTION],
    "Expense": [ExpenseAccount, EXPENSE_COLLECTION],
}
router = APIRouter(prefix="/api/v1", tags=["Import Bank Account"])

def create_new_account(user_id: str, bank_id: str, number_of_account: int = 0):
    account_generic = AccountGeneric(
        user_id=ObjectId(user_id),
        bank_id=ObjectId(bank_id),
        number_of_account=number_of_account,
    ).dict()
    ACCOUNT_COLLECTION.insert_one(account_generic)
    return account_generic["_id"]

def verify_account(user_id: str,bank_id: str) ->bool:
    query = {"bank_id": ObjectId(bank_id), "user_id": ObjectId(user_id)}
    account_exist = ACCOUNT_COLLECTION.find_one(query)
    if account_exist:
        ACCOUNT_COLLECTION.update_one(query, {"$inc": {"number_of_account": 1}})
        return True
    else: 
        return False

def verify_account_info(
    account_name: str,
    account_type_model: list[BaseAccount, MongoClient],
    account_id: ObjectId,
    user_id: ObjectId,
    bank_id: ObjectId,
    current_balance: float,
) -> bool:
    if (
        account_type_model[1].find_one(
            {"account_name": account_name, "bank_id": bank_id}
        )
        is None
    ):
        account_type_model[1].insert_one(
            account_type_model[0](
                account_name=account_name,
                account_id=account_id,
                user_id=user_id,
                bank_id=bank_id,
                goal_id=None,
                current_balance=current_balance,
            ).dict()
        )
        query = {"bank_id": bank_id, "user_id": user_id}
        ACCOUNT_COLLECTION.update_one(query, {"$inc": {"number_of_account": 1}})
        return True
    else:
        return False
      


@router.post('/{user_id}/bank_account_import')
def bank_account_import(user_id: str, csv_file: UploadFile = File(...)):
    headers = ["account_name", "account_type", "current_balance", "bank_name"]
    df = pd.read_csv(BytesIO(csv_file.file.read()), names=headers, skiprows= 1)
    data = df.to_dict(orient='records')
    for value in data:
        if value["account_type"] not in MODEL:
            return JSONResponse(
            status_code=422, content={"message": f"Invalid account type:{value} /n {value['account_type']}"}
        )
        bank = BANK_COLLECTION.find_one({"bank_name": value["bank_name"]})
        if bank is None:
            return JSONResponse(status_code=422, content={ "message": f"Invalid bank name: {value}"})
        bank_id = bank["_id"]
        account = ACCOUNT_COLLECTION.find_one({"bank_id": ObjectId(bank_id), "user_id": ObjectId(user_id)})
        if account is None:
            account_id = create_new_account(user_id=user_id, bank_id=bank_id)
        else:
            account_id = account["_id"]
        created_account = verify_account_info(
            account_name=value["account_name"],
            account_type_model=MODEL[value["account_type"]],
            account_id=ObjectId(account_id),
            user_id=ObjectId(user_id),
            bank_id=ObjectId(bank_id),
            current_balance= value["current_balance"],
        )
        if created_account == False:
            return JSONResponse(status_code=422, content={"message": "Bank account already exists"})
    return JSONResponse( content= data)