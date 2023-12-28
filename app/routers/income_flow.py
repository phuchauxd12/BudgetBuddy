from fastapi import APIRouter
from models.income_flow import Bank
from bson import ObjectId
from pymongo import MongoClient
from constant import Constant


bank = APIRouter(prefix="/api/v1")
connection = MongoClient()
CLIENT = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
db = CLIENT.get_collection('IncomeFlow')
collection = db["IncomeFlowTest"]

def income_flow(item) -> dict:
    return  {
        'id': str(item["_id"]),
        'date': str('date'),
        'total_income': str('total_income'),
        'total_expenses': str('total_expenses')
    }
    
def banksEntity(items)-> list:
    return[income_flow(item) for item in items ]


@bank.get('/get_income_data')
async def find_all_bank():
    return banksEntity(collection.find())

@bank.post('/input_income_data')
async def create_bank(bank: Bank):
    """ 
    Allow use to manually input there income flow transaction 
    date: Datetime
    total_income: Float
    total_expenses
    These informaitons are used for user income prediction"""
    collection.insert_one(dict(bank))
    return banksEntity(collection.find())    

@bank.get('/get_by_id/{id}')
async def find_all_bank():
    collection.find_one(dict(bank))
    return banksEntity(collection.find())

@bank.put('/update_by_id/{id}')
async def update_bank(id: str, bank: Bank):
    collection.find_one_and_update({"_id": ObjectId(id)},{"$set":dict(bank)})

@bank.delete('/delete_by_id/{id}')
async def delete_bank(id):
    collection.find_one_and_delete({"_id": ObjectId(id)})
