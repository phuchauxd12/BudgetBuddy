from fastapi import APIRouter
from models.income_flow import Bank
from bson import ObjectId
from pymongo import MongoClient
from constant import Constant


bank = APIRouter(prefix="/api/v1")
connection = MongoClient()
CLIENT = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
collection = CLIENT.get_collection('IncomeFlow')


def income_flow(item) -> dict:
    return  {
        'income_id': str(item["_id"]),
        'date':str(item['date']),
        'total_income': item['total_income'],
        'total_expenses': item['total_expenses']
    }


def banksEntity(items)-> list:
    return[income_flow(item) for item in items ]


@bank.get('/get_income_data')
async def find_all_income():
    return banksEntity(collection.find())

@bank.post('/input_income_data')
async def create_income_flow(bank: Bank):
    collection.insert_one(dict(bank))
    return banksEntity(collection.find())


@bank.get('/get_by_id/{id}')
async def find_by_id(income_id: str):
    existing_id = income_flow(collection.find_one({"_id": ObjectId(income_id)}))
    return existing_id

@bank.put('/update_by_id/{id}')
async def update_income(income_id: str, bank: Bank):
    return collection.find_one_and_update({"_id": ObjectId(income_id)},{"$set":dict(bank)})

@bank.delete('/delete_by_id/{id}')
async def delete_income(income_id: str):
    return collection.find_one_and_delete({"_id": ObjectId(income_id)})
