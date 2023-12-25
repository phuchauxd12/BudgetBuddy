from fastapi import APIRouter
from bank_models import Bank
from db import connection
from bank_schema import bankEntity, banksEntity
bank = APIRouter()

@bank.get('/')
async def find_all_bank():
    print(connection.local.bank.find())
    print(banksEntity(connection.local.bank.find()))
    return banksEntity(connection.local.bank.find())

@bank.post('/')
async def create_bank(bank: Bank):
    connection.local.bank.find(dict(bank))
    return banksEntity(connection.local.bank.find())    
