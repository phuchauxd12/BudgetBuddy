from __future__ import annotations

import yfinance as yf

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from constant import Constant
from bson import ObjectId
from models.investment_model import Stock, ListOfStocks, GetInvestedAccountData

router = APIRouter(prefix="/api/v1", tags=["Investment Page"])

STOCK_LIST = ["aapl", "msft", "tsla", "metv", "goog", "amzn", "nflx"]
CLIENT = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
INVESTMENT_COLLECTION = CLIENT.get_collection("INVESTMENT_ACCOUNTS")

@router.get("/get_all_stocks", response_model=ListOfStocks)
def get_stocks():
    stock_list = []
    for name in STOCK_LIST:
        stock = yf.Ticker(name)
        previous_price = stock.info['previousClose']
        current_price = stock.info['open']
        price_difference = round(current_price - previous_price, 2)
        stock_list.append(Stock(
            stock_symbol=name.upper(),
            stock_name=stock.info['shortName'],
            current_price=current_price,
            previous_close=previous_price,
            price_diff=price_difference
        ))
    
    return ListOfStocks(list_of_stocks=stock_list)

@router.get("/get_invested_account_data/{user_id}", response_model=GetInvestedAccountData)
def get_invested_account_data(user_id: str):
    account_data = INVESTMENT_COLLECTION.find({"user_id": ObjectId(user_id)})
    return JSONResponse(status_code=200, content=[GetInvestedAccountData(**account_data).dict() for account_data in account_data])