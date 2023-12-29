# -*- coding: utf-8 -*-
from __future__ import annotations

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from routers import (
    register, newsletter,
    bank_account, bank_info,
    profile, login, goal,
    transaction, bank_account_import,
    user_bills, plan_spending, stock)


app = FastAPI(
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register.router)
app.include_router(plan_spending.router)
app.include_router(stock.router)
app.include_router(newsletter.router)
app.include_router(bank_account.router)
app.include_router(bank_info.router)
app.include_router(login.router)
app.include_router(profile.router)
app.include_router(goal.router)
app.include_router(transaction.router)
app.include_router(bank_account_import.router)
app.include_router(user_bills.router)
if __name__ == "__main__":
    uvicorn.run("main:app", workers=1, host="0.0.0.0", port=8080)
