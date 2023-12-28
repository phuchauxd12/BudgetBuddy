# -*- coding: utf-8 -*-
from __future__ import annotations

from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Optional


class GoalSettingBaseModel(BaseModel):
    user_id: ObjectId = Field(..., description="User id")
    goal_created_date: str = Field(..., description="Goal created date")
    goal_name: str = Field(..., description="Goal name")
    goal_end_date: str = Field(None, description="Goal end date")
    saving_amount: float | int = Field(..., description="Goal value")
    connected_account_name: Optional[str] = Field(None, description="Account id")
    connected_account_id: Optional[ObjectId] = Field(None, description="Account id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class GoalModelView(BaseModel):
    goal_created_date: str
    goal_name: str
    goal_end_date: Optional[str]
    saving_amount: float | int