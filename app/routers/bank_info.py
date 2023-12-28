from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from constant import Constant
from pymongo import MongoClient
from models.bank_info import BankInfo, ListOfBankInfo

router = APIRouter(prefix="/api/v1", tags=["Banking Data Info"])
client = MongoClient(host= Constant.MONGODB_URI ).get_database("dev")
bank_info = client.get_collection("BANK_INFO")

@router.get("/get_bank_data")
def get_bank_data():
    response = []
    for data in bank_info.find({}):
        response.append(BankInfo(
            bank_name=data['bank_name'],
            bank_head_quarter_address=data['bank_head_quarter_address'],
            bank_swift_code=data['bank_swift_code']
        )
    )
    return ListOfBankInfo(list_of_bank=response)

@router.post("/create_bank_info")
def create_bank_data(
    bank_name: str = Form(..., description="Name of the bank"),
    bank_head_quarter_address: str = Form(..., description="Head quarter address of the bank"),
    bank_swift_code: str = Form(..., description="Swift code of the bank")
):
    bank_info.insert_one(
        BankInfo(
            bank_name=bank_name,
            bank_head_quarter_address=bank_head_quarter_address,
            bank_swift_code=bank_swift_code
        ).dict()
    )
    
    return JSONResponse(content={"message": "Bank data created successfully"})