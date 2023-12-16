# -*- coding: utf-8 -*-
from __future__ import annotations

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import register, newsletter

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
app.include_router(newsletter.router)

if __name__ == "__main__":
    uvicorn.run("main:app", workers=1, host="0.0.0.0", port=8080, reload= True)
