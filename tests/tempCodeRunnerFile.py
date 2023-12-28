import bcrypt
# from app.constant import Constant



# def test_create_user():
#     client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
#     db = client.get_collection("USERS")
#     user_name = "tuanxautrai"
#     password = "123"
#     full_name = "shitboy"
#     number = "1234567899"
#     email = "mail@gmail.com"
#     address = "ko co"
#     stored_data = db.find_one({'username': user_name})
#     assert stored_data is not None
#     assert stored_data["username"] == user_name
#     assert bcrypt.checkpw(password.encode('utf-8'), stored_data["password"].encode('utf-8'))
#     assert stored_data["full_name"] == full_name
#     assert stored_data["number"] == number
#     assert stored_data["email"] == email
#     assert stored_data["address"] == address

# def test_user_aldready_exists():
#     client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
#     db = client.get_collection("USERS")
#     user_name = "tuanxautrai"
#     password = "123"
#     full_name = "shitboy1223"
#     number = "1233267899"
#     email = "ma23il@gmail.com"
#     stored_data = db.find({'username': user_name})
#     assert stored_data is not None
#     assert len(list(stored_data)) == 1

# def test_wrong_data_type():
#     client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
#     db = client.get_collection("USERS")
#     user_name = "tuanwibu"
#     password = "123"
#     full_name = "shitboy1223"
#     number = "123asdw"
#     email = "ma23il.com"
#     assert len(list(db.find({'username': user_name}))) == 0
#     assert len(list(db.find({'email': email}))) == 0
#     assert len(list(db.find({"number": number}))) == 0


# def test_missing_data():
#     client = MongoClient(host=Constant.MONGODB_URI).get_database("dev")
#     db = client.get_collection("USERS")
#     user_name = "tuanwibu"
#     password = "123"
#     full_name = "shitboy1223"
#     email = None
#     assert len(list(db.find({'username': user_name}))) == 0
#     assert len(list(db.find({'email': email}))) == 0
