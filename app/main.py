# -*- coding: utf-8 -*-
from __future__ import annotations

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from routers import (
    register, newsletter,
    bank_account,
    profile, login, goal,
    transaction,
    user_bills, plan_spending, investment_api, report_api, prediction)


app = FastAPI(
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
)
origins = [
    "http://localhost:8081",  # Replace with the actual origin of your Vue.js app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register.router)
app.include_router(plan_spending.router)
app.include_router(investment_api.router)
app.include_router(newsletter.router)
app.include_router(bank_account.router)
app.include_router(login.router)
app.include_router(profile.router)
app.include_router(goal.router)
app.include_router(transaction.router)
app.include_router(user_bills.router)
app.include_router(report_api.router)
app.include_router(prediction.router)
if __name__ == "__main__":
    uvicorn.run("main:app", workers=1, host="0.0.0.0", port=8080)
