from fastapi import FastAPI
from route.bank_route import bank

app = FastAPI()
app.include_router(bank)