from fastapi import APIRouter
from models.bank_models import Bank
from db.db import connection
from schema.bank_schema import bankEntity, banksEntity
from bson import objectid

bank = APIRouter()

@bank.get('/')
async def find_all_bank():
    print(connection.local.test.find())
    print(banksEntity(connection.local.test.find()))
    return banksEntity(connection.local.test.find())

@bank.post('/')
async def create_bank(bank: Bank):
    connection.local.test.find(dict(bank))
    return banksEntity(connection.local.test.find())    

@bank.put('/{id}')
async def update_bank(id, bank: Bank):
    connection.local.test.find_one_and_update({"_id": objectid(id)},{
        "$set":dict(bank)
    })
    return bankEntity(connection.local.test.find({"_id":objectid(id)}))    

@bank.delete('/{id}')
async def delete_bank(id, bank: Bank):
    return bankEntity(connection.local.test.find_one_and_delete({"_id": objectid(id)}))
