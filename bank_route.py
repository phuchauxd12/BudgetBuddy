from fastapi import APIRouter
from bank_models import Bank
from db import connection
from bank_schema import bankEntity, banksEntity
from bson import objectid
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

@bank.put('/{id}')
async def update_bank(id, bank: Bank):
    connection.local.bank.find_one_and_update({"_id": objectid(id)},{
        "$set":dict(bank)
    })
    return bankEntity(connection.local.bank.find({"_id":objectid(id)}))    

@bank.delete('/{id}')
async def delete_bank(id, bank: Bank):
    return bankEntity(connection.local.bank.find_one_and_delete({"_id": objectid(id)}))
