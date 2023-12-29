from pydantic import BaseModel, Field

class Stock(BaseModel):
    stock_symbol: str = Field(..., description = "The stock symbol")
    stock_name: str = Field(..., description = "The stock name")
    current_price: float = Field(..., description = "The current price of the stock")
    previous_close: float = Field(..., description = "The previous close of the stock")
    price_diff: float = Field(..., description = "The price difference of the stock")

class ListOfStocks(BaseModel):
    list_of_stocks: list[Stock] = []

class GetInvestedAccountData(BaseModel):
    account_name: str = Field(..., description="Account name of the bank account")
    current_balance: float = Field(..., description="Invested amount of the bank account")