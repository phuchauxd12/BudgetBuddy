from fastapi import FastAPI
from client.bank_route import bank

app = FastAPI()
app.include_router(bank)