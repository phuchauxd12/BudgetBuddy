from pymongo import MongoClient
from pymongo import MongoClient
connection = MongoClient("mongodb+srv://userDB:InC2QuunWeQUFOCm@userdb.opquo83.mongodb.net/")
db = connection.UserBank
collection = db["BankTest"]