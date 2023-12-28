import pytest
from starlette.testclient import TestClient
from fastapi import FastAPI
from app.routers import bank_account
from unittest.mock import patch, MagicMock
from fastapi.responses import JSONResponse

app = FastAPI()
app.include_router(bank_account.router)

client = TestClient(app)

@patch('routers.bank_account.bank_collection')
@patch('routers.bank_account.account_collection')
def test_create_account_manually(mock_account_collection, mock_bank_collection):
    mock_bank_collection.find_one.return_value = {"_id": "123"}
    mock_account_collection.find_one.return_value = None

    response = client.post(
        "/api/v1/bank_account/123/create_manually",
        data={
            "account_name": "Test Account",
            "account_type": "Saving",
            "current_balance": 1000.0,
            "bank_name": "Test Bank",
        }
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Bank account created successfully"}

    mock_bank_collection.find_one.assert_called_once_with({"bank_name": "Test Bank"})
    mock_account_collection.find_one.assert_called_once_with({"bank_id": "123", "user_id": "123"})