from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from constant import Message, USERS
from pymongo import MongoClient
import bcrypt

router = APIRouter(prefix="/api/v1", tags=["User Login"])

@router.post('/login', responses={401: {'model': Message}, 404: {'model': Message}})
def login(user_name: str = Form(..., description="Username of the user"),
          password: str = Form(..., description="Password of the user"),
          key: str = Form(..., description='Key for validation')
          ):

    # Find the user account in the database
    account = USERS.find_one({'username': user_name})
    
    if account:
        # Compare the entered password with the stored hashed password
        if bcrypt.checkpw(password.encode('utf-8'), account['password'].encode('utf-8')):
            if key == account['key']:
                return JSONResponse(content={"user_name": user_name, "user_id": str(account["_id"])})
            else:
                return JSONResponse(status_code=401, content={'message': "Key is incorrect."})
        else:
            return JSONResponse(status_code=401, content={'message': "Password is incorrect."})
    else:
        return JSONResponse(status_code=404, content={'message': "User does not exist."})