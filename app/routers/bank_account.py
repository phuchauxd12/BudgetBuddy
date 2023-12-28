from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from constant import Constant
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

# MONGODB CONNECTION
CLIENT = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
BANK_COLLECTION = CLIENT.get_collection("BANK_INFO")
ACCOUNT_COLLECTION = CLIENT.get_collection("ACCOUNTS")
SAVING_COLLECTION = CLIENT.get_collection("SAVING_ACCOUNTS")
INVESTMENT_COLLECTION = CLIENT.get_collection("INVESTMENT_ACCOUNTS")
EXPENSE_COLLECTION = CLIENT.get_collection("EXPENSE_ACCOUNTS")

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


@router.get(
    "bank_account/{user_id}/{account_type}/{account_name}", response_model=GetAccountInformation
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

    bank_id = BANK_COLLECTION.find_one({"_id": account_data["bank_id"]})["bank_name"]

    return GetAccountInformation(
        account_name=account_data["account_name"],
        account_type=account_type,
        bank_name=bank_id,
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
    account_id = ACCOUNT_COLLECTION.find_one({"user_id": ObjectId(user_id)})["_id"]

    result = []
    for key, value in MODEL.items():
        data = value[1].find_one({"account_id": ObjectId(account_id)})
        if data is not None:
            result.append([data["account_name"], data["current_balance"]])

    return GetAllAccountName(list_account_name=result)

@router.put("/{user_id}/{account_type}/{account_name}")
def update_account_info(
    user_id: str, account_type: str, account_name: str, new_value: dict[str, str]
):
    """
    Update the information of a bank account.

    Args:
        user_id (str): The ID of the user.
        account_type (str): The type of the account.
        account_name (str): The name of the account.
        new_value (dict[str, str]): The new values to update the account with.

    Returns:
        JSONResponse: The response indicating the success of the update.
    """
    account_type_model = MODEL[account_type]

    if "account_name" in new_value:
        account_type_model[1].update_one(
            {"account_name": account_name, "user_id": ObjectId(user_id)},
            {"$set": {"account_name": new_value["account_name"]}},
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
