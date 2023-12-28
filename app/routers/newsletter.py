from __future__ import annotations
from functools import lru_cache

from fastapi import APIRouter
import yfinance as yf
from datetime import datetime

from constant import Message
from models.news import News, ListOfNews

router = APIRouter(prefix="/api/v1", tags=["News Feed"])
STOCK_LIST = ["aapl", "msft", "tsla"]


def flatten(nested_list):
    """
    Flatten a nested list into a single list.

    Args:
        nested_list (list): A nested list containing any number of nested lists.

    Returns:
        list: A single list containing all the items from the nested list.
    """
    list_of_lists = []
    for item in nested_list:
        list_of_lists.extend(item)
    return list_of_lists


def get_tickers() -> yf.Tickers:
    """
    Returns a yfinance.Tickers object for the global list of stock tickers.
    """
    return yf.Tickers(" ".join(STOCK_LIST))

def get_ticker_news(tickers: yf.Tickers, ticker_name: str) -> list[dict]:
    """
    Given a yfinance.Tickers object and a ticker name, returns a list of news items for that ticker.

    Args:
        tickers (yf.Tickers): A yfinance.Tickers object containing multiple tickers.
        ticker_name (str): The name of the ticker for which news items are to be retrieved.

    Returns:
        list[dict]: A list of dictionaries representing news items for the specified ticker.
    """
    return [news for news in tickers.tickers[ticker_name.upper()].news if "thumbnail" in list(news.keys())]

def create_news_objects(ticker_news: list[dict]) -> list[News]:
    """
    Given a list of news items for a ticker, returns a list of News objects.

    Args:
        ticker_news (list[dict]): A list of news items for a ticker.

    Returns:
        list[News]: A list of News objects.
    """
    return [
        News(
            news_title=news.get("title"),
            news_url=news.get("link"),
            news_image_url=news["thumbnail"]["resolutions"][0]["url"],
            update_date=datetime.utcfromtimestamp(int(news.get("providerPublishTime"))).strftime("%Y-%m-%d"),
            publisher=news.get("publisher"),
        )
        for news in ticker_news
    ]

def get_news_yahoo() -> list[list[News]]:
    """
    Returns a list of lists of News objects for each ticker in the global list of stock tickers.

    Returns:
        list[list[News]]: A list of lists of News objects.
    """
    tickers = get_tickers()
    all_news = [get_ticker_news(tickers, ticker_name) for ticker_name in STOCK_LIST]
    list_of_news = [create_news_objects(ticker_news) for ticker_news in all_news]
    return list_of_news



@router.get("/get_news", responses = {404: {"model": Message}})
@lru_cache()
def get_news():
    result = get_news_yahoo()
    return ListOfNews(list_of_news=flatten(result))
