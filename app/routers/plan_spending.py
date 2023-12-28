from __future__ import annotations

from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from bson import ObjectId
from constant import Constant
from pymongo import MongoClient

from models.plan_spending import PlannedSpendingModel, PlannedSpendingModelView

router = APIRouter(prefix="/api/v1", tags=["Plan Settings"])

CLIENT = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
PLANNING_SPENDING_COLLECTION = CLIENT.get_collection("PLANNING_SPENDING")


@router.post("/plan_spending/{user_id}/create")
def create_spending(
    user_id: str,
    spending_name: str = Form(..., description="Name of the spending"),
    amount: int | float = Form(..., description="Amount of the spending"),
    spending_type: str = Form(..., description="Type of the spending. This should either be Income, Bills, Subscription or Credit Card Payment"),
    time_duration: str = Form(..., description="Time duration of the spending planning. Example: Jan 2024, Feb 2024, etc.")
):
    if PLANNING_SPENDING_COLLECTION.find_one({"user_id": ObjectId(user_id), "spending_name": spending_name}) is not None:
        raise HTTPException(status_code=400, detail="Spending name already exist")
    
    spending = PlannedSpendingModel(
        spending_name=spending_name,
        amount=amount,
        user_id=ObjectId(user_id),
        spending_type=spending_type,
        time_duration=time_duration
    ).dict()
    PLANNING_SPENDING_COLLECTION.insert_one(spending)
    return JSONResponse(status_code=200, content={"message": "Spending created successfully"})

@router.get("/plan_spending/{user_id}/{timeframe}")
def get_all_spending(user_id: str, timeframe: str):
    spending = PLANNING_SPENDING_COLLECTION.find({"user_id": ObjectId(user_id), "time_duration": timeframe})
    return JSONResponse(
        status_code=200,
        content=[PlannedSpendingModelView(**spending).dict() for spending in spending])

@router.put("/plan_spending/{user_id}/{spending_id}")
def update_spending(
    user_id: str,
    spending_id: str,
    spending_name: str = Form(..., description="Name of the spending"),
    amount: int | float = Form(..., description="Amount of the spending"),
    spending_type: str = Form(..., description="Type of the spending. This should either be Income, Bills, Subscription or Credit Card Payment"),
    time_duration: str = Form(..., description="Time duration of the spending planning. Example: Jan 2024, Feb 2024, etc.")
):
    spending = PLANNING_SPENDING_COLLECTION.find_one({"_id": ObjectId(spending_id)})
    if spending is None:
        raise HTTPException(status_code=400, detail="Spending does not exist")
    
    PLANNING_SPENDING_COLLECTION.update_one(
        {"_id": ObjectId(spending_id)},
        {"$set": {
            "spending_name": spending_name,
            "amount": amount,
            "spending_type": spending_type,
            "time_duration": time_duration
        }}
    )
    return JSONResponse(status_code=200, content={"message": "Spending updated successfully"})

@router.delete("/plan_spending/{user_id}/{spending_id}")
def delete_spending(user_id: str, spending_id: str):
    spending = PLANNING_SPENDING_COLLECTION.find_one({"_id": ObjectId(spending_id)})
    if spending is None:
        raise HTTPException(status_code=400, detail="Spending does not exist")
    
    PLANNING_SPENDING_COLLECTION.delete_one({"_id": ObjectId(spending_id), })
    return JSONResponse(status_code=200, content={"message": "Spending deleted successfully"})    