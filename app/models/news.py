from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field

class News(BaseModel):
    """Financial news information from Yahoo Finance API Model

    Args:
        BaseModel (_type_): the base model of the news
    """
    
    news_title: str = Field(..., description="Username of the user")
    news_url: str = Field(..., description="Username of the user")
    news_image_url: str = Field(..., description="Username of the user")
    update_date: Optional[str] = Field(..., description="Username of the user")
    publisher: str = Field(..., description="The publisher of the news")

class ListOfNews(BaseModel):
    list_of_news: list[News] = []