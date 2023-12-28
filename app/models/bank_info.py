from __future__ import annotations

from pydantic import BaseModel, Field

class BankInfo(BaseModel):
    bank_name: str = Field(..., description="Name of the bank")
    bank_head_quarter_address: str = Field(..., description="Head quarter address of the bank")
    bank_swift_code: str = Field(..., description="Swift code of the bank")

class ListOfBankInfo(BaseModel):
    list_of_bank: list[BankInfo] = []