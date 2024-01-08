##from enum import StrEnum
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from enum import Enum

import os
import certifi as certifi
class StrEnum(str, Enum):
    pass

# Load environment variables from the local.env file
load_dotenv(os.path.join(os.path.dirname(__file__), "local.env"))

class Constant(StrEnum):
    MONGODB_URI =  'mongodb+srv://phuchauxd12:Abcd0123@cluster0.lf8sh9p.mongodb.net/'



class Message(BaseModel):
    message: str

client = MongoClient(host=Constant.MONGODB_URI, tlsCAFile=certifi.where(), tls=True).get_database("dev")
BANK_COLLECTION = client.get_collection("BANK_INFO")
ACCOUNT_COLLECTION = client.get_collection("ACCOUNTS")
SAVING_COLLECTION = client.get_collection("SAVING_ACCOUNTS")
INVESTMENT_COLLECTION = client.get_collection("INVESTMENT_ACCOUNTS")
EXPENSE_COLLECTION = client.get_collection("EXPENSE_ACCOUNTS")
BANK_INFO = client.get_collection("BANK_INFO")
GOAL_SETTING_BASE = client.get_collection("GOAL_SETTINGS")
USERS = client.get_collection("USERS")
PLANNING_SPENDING_COLLECTION = client.get_collection("PLANNING_SPENDING")
EXPENSE_SPENDING_COLLECTION = client.get_collection("EXPENSE_SPENDING")
TRANSACTION_COLLECTION = client.get_collection("TRANSACTION_HISTORY")
GOAL_SETTINGS = client.get_collection("GOAL_SETTINGS")
USER_BILLS = client.get_collection("USER_BILLS")