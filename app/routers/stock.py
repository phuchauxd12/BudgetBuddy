from __future__ import annotations

import yfinance as yf

from fastapi import APIRouter
from models.stock import Stock, ListOfStocks

router = APIRouter(prefix="/api/v1", tags=["Stock Data"])
STOCK_LIST = ["aapl", "msft", "tsla", "metv", "goog", "amzn", "nflx"]

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