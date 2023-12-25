from fastapi import FastAPI
from bank_route import bank

app = FastAPI()
app.include_router(bank)