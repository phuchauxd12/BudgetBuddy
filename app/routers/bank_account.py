from __future__ import annotations

from fastapi import APIRouter, Form, File, UploadFile
import pandas as pd
from io import BytesIO
from fastapi.responses import JSONResponse
from constant import BANK_COLLECTION, ACCOUNT_COLLECTION, SAVING_COLLECTION, INVESTMENT_COLLECTION, EXPENSE_COLLECTION
from pymongo import MongoClient
from models.bank_account import (
    SavingOrInvestmentAccount,
    ExpenseAccount,
    AccountGeneric,
    BaseAccount,
    GetAccountInformation,
    GetAllAccountName,
)
from bson import ObjectId

router = APIRouter(prefix="/api/v1", tags=["CRUD Bank Account"])


# MODEL INITIALIZATION
MODEL = {
    "Saving": [SavingOrInvestmentAccount, SAVING_COLLECTION],
    "Investment": [SavingOrInvestmentAccount, INVESTMENT_COLLECTION],
    "Expense": [ExpenseAccount, EXPENSE_COLLECTION],
}


def create_new_account_generic(user_id: str, bank_id: str, number_of_account: int = 0):
    """
    Create a new generic account.

    Args:
        user_id (str): The ID of the user.
        bank_id (str): The ID of the bank.
        number_of_account (int, optional): The number of accounts. Defaults to 0.

    Returns:
        str: The ID of the created account.
    """
    account_generic = AccountGeneric(
        user_id=ObjectId(user_id),
        bank_id=ObjectId(bank_id),
        number_of_account=number_of_account,
    ).dict()
    ACCOUNT_COLLECTION.insert_one(account_generic)
    return account_generic["_id"]


def verify_account_info(
    account_name: str,
    account_type_model: list[BaseAccount, MongoClient],
    account_id: ObjectId,
    user_id: ObjectId,
    bank_id: ObjectId,
    current_balance: float,
) -> bool:
    """
    Verifies the account information and inserts a new account if it doesn't exist.

    Args:
        account_name (str): The name of the account.
        account_type_model (list[BaseAccount, MongoClient]): The account type model.
        account_id (ObjectId): The ID of the account.
        user_id (ObjectId): The ID of the user.
        bank_id (ObjectId): The ID of the bank.
        current_balance (float): The current balance of the account.

    Returns:
        bool: True if the account is verified and inserted successfully, False otherwise.
    """
    if (
        account_type_model[1].find_one(
            {"account_name": account_name, "bank_id": bank_id,  "user_id": user_id},
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


# create account manually
@router.post("/bank_account/{user_id}/create_manually")
def create_account_manually(
    user_id: str,
    account_name: str = Form(..., description="Account name of the bank account"),
    account_type: str = Form(..., description="Account type of the bank account"),
    current_balance: float = Form(..., description="Current balance of the bank account"),
    bank_name: str = Form(..., description="Bank name of the bank account"),
) -> JSONResponse:
    """
    Create a bank account manually.

    Args:
        user_id (str): The ID of the user.
        account_name (str): Account name of the bank account.
        account_type (str): Account type of the bank account.
        current_balance (float): Current balance of the bank account.
        bank_name (str): Bank name of the bank account.

    Returns:
        JSONResponse: The response containing the status and message.

    Raises:
        HTTPException: If the account type is invalid or the bank name is not available.

    """
    # Check if account type is valid. Must be either Saving, Investment, or Expense
    if account_type not in MODEL:
        return JSONResponse(
            status_code=422, content={"message": "Invalid account type"}
        )

    # Check if bank name is available in the database
    bank = BANK_COLLECTION.find_one({"bank_name": bank_name})
    if bank is None:
        return JSONResponse(status_code=422, content={"message": "Invalid bank name"})

    bank_id = bank["_id"]

    account = ACCOUNT_COLLECTION.find_one(
        {"bank_id": ObjectId(bank_id), "user_id": ObjectId(user_id)}
    )
    if account is None:
        account_id = create_new_account_generic(user_id=user_id, bank_id=bank_id)
    else:
        account_id = account["_id"]

    created_account = verify_account_info(
        account_name=account_name,
        account_type_model=MODEL[account_type],
        account_id=ObjectId(account_id),
        user_id=ObjectId(user_id),
        bank_id=ObjectId(bank_id),
        current_balance=current_balance,
    )

    if created_account:
        return JSONResponse(
            status_code=200, content={"message": "Bank account created successfully"}
        )
    else:
        return JSONResponse(
            status_code=422, content={"message": "Bank account already exists"}
        )

@router.post('/bank_account/{user_id}/bank_account_import')
def bank_account_import(user_id: str, csv_file: UploadFile = File(...)):
    headers = ["account_name", "account_type", "current_balance", "bank_name"]
    df = pd.read_csv(BytesIO(csv_file.file.read()), names=headers, skiprows= 1)
    data = df.to_dict(orient='records')
    for value in data:
        if value["account_type"] not in MODEL:
            return JSONResponse(
            status_code=422, content={"message": f"Invalid account type:{value}"}
        )
        bank = BANK_COLLECTION.find_one({"bank_name": value["bank_name"]})
        if bank is None:
            return JSONResponse(status_code=422, content={ "message": f"Invalid bank name: {value}"})
        bank_id = bank["_id"]
        account = ACCOUNT_COLLECTION.find_one({"bank_id": ObjectId(bank_id), "user_id": ObjectId(user_id)})
        if account is None:
            account_id = create_new_account_generic(user_id=user_id, bank_id=bank_id)
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

@router.get(
    "/bank_account/{user_id}/{account_type}/{account_name}", response_model=GetAccountInformation
)
def get_bank_account_info(
    user_id: str,
    account_type: str,
    account_name: str,
):
    """
    Retrieve information about a bank account.

    Args:
        user_id (str): The ID of the user.
        account_type (str): The type of the account.
        account_name (str): The name of the account.

    Returns:
        GetAccountInformation: The account information.

    Raises:
        None.
    """
    account_type_model = MODEL[account_type]
    account_data = account_type_model[1].find_one(
        {"account_name": account_name, "user_id": ObjectId(user_id)}
    )

    bank_name = BANK_COLLECTION.find_one({"_id": account_data["bank_id"]})["bank_name"]

    return GetAccountInformation(
        account_name=account_data["account_name"],
        account_type=account_type,
        bank_name=bank_name,
        current_balance=account_data["current_balance"],
    )

@router.get("/{user_id}")
def get_all_account_data(user_id: str):
    """
    Retrieve all account name of a user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        GetAllAccountName: The list of account name of the user.

    Raises:
        None.
    """
    list_account_id = [account["_id"] for account in ACCOUNT_COLLECTION.find({"user_id": ObjectId(user_id)})]

    result = []
    for key, value in MODEL.items():
            for account_id in list_account_id:
                list_data = value[1].find({"account_id": ObjectId(account_id)})
                if list_data:
                    for data in list_data:
                        bank_name = BANK_COLLECTION.find_one({"_id": ObjectId(data["bank_id"])})["bank_name"]
                        result.append([data["account_name"], data["current_balance"], 
                                       str(ObjectId(data["account_id"])),bank_name, key])

    return GetAllAccountName(list_account_name=result)

@router.get(
    "/bank_account/{user_id}/{account_type}"
)
def get_all_account_type(
    user_id: str,
    account_type: str,
):
    account_type_model = MODEL[account_type]
    list_account = account_type_model[1].find(
        {"user_id": ObjectId(user_id)}
    )
    
    result = []
    for account in list_account:
        bank_name = BANK_COLLECTION.find_one({"_id": account["bank_id"]})["bank_name"]
        account_data = GetAccountInformation(**account, bank_name= bank_name).dict()
        account_data["id"] = str(account["_id"])
        result.append(account_data) 
    return JSONResponse(status_code=200,content= result)

@router.put("/{user_id}/{account_type}/{account_name}")
def update_account_info(
    user_id: str, account_type: str, account_name: str, 
    new_account_name: str = Form(None, description="Account name of the user"),
    new_bank_name:  str = Form(None, description="New bank name"),
):
    """
    Update the information of a bank account.

    Args:
        user_id (str): The ID of the user.
        account_type (str): The type of the account.
        account_name (str): The name of the account.

    Returns:
        JSONResponse: The response indicating the success of the update.
    """
    account_type_model = MODEL[account_type]
    update_data = {}
    if new_account_name is not None:
        update_data["account_name"]= new_account_name
    if new_bank_name is not None:
        bank = BANK_COLLECTION.find_one({"bank_name": new_bank_name})
        if bank is None:
            return JSONResponse(status_code=422, content={ "message": f"Invalid bank name"})
        bank_id = bank["_id"]
        update_data["bank_id"] = ObjectId(bank_id)
    account_type_model[1].update_one(
            {"account_name": account_name, "user_id": ObjectId(user_id)},
            {"$set": update_data},
        )

    return JSONResponse(
        status_code=200, content={"message": "Bank account updated successfully"}
    )


@router.delete("/{user_id}/{account_type}/{account_name}")
def delete_account(
    user_id: str,
    account_type: str,
    account_name: str,
):
    """
    Delete a bank account for a specific user.

    Args:
        user_id (str): The ID of the user.
        account_type (str): The type of the account.
        account_name (str): The name of the account.

    Returns:
        JSONResponse: The response indicating the success of the deletion.
    """
    account_type_model = MODEL[account_type]
    account_type_model[1].delete_one(
        {"account_name": account_name, "user_id": ObjectId(user_id)}
    )
    ACCOUNT_COLLECTION.update_one(
        {"user_id": ObjectId(user_id)}, {"$inc": {"number_of_account": -1}}
    )
    return JSONResponse(
        status_code=200, content={"message": "Bank account deleted successfully"}
    )