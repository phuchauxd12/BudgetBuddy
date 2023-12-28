from fastapi import APIRouter
from app.models.bank_models import Bank
from app.db.db import connection, db, collection
from app.schema.bank_schema import bankEntity, banksEntity
from bson import objectid

bank = APIRouter()

@bank.get('/')
async def find_all_bank():
    print(collection.test.find())
    print(banksEntity(collection.test.find()))
    return banksEntity(collection.find())


@bank.post('/')
async def create_bank(bank: Bank):
    collection.insert_one(dict(bank))
    return banksEntity(collection.find())    

@bank.put('/{id}')
async def update_bank(id, bank: Bank):
    collection.find_one_and_update({"_id": objectid(id)},{
        "$set":dict(bank)
    })
    return bankEntity(collection.find({"_id":objectid(id)}))    

@bank.delete('/{id}')
async def delete_bank(id, bank: Bank):
    return bankEntity(collection.find_one_and_delete({"_id": objectid(id)}))
