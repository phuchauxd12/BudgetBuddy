from app.models.news import News, ListOfNews
from app.routers.newsletter import flatten, get_tickers, get_ticker_news, create_news_objects, get_news_yahoo, get_news
import yfinance as yf



def test_flatten():
    nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected_result = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flatten(nested_list) == expected_result

def test_get_tickers():
    tickers = get_tickers()
    assert isinstance(tickers, yf.Tickers)

def test_get_ticker_news():
    tickers = get_tickers()
    ticker_name = "AAPL"
    news = get_ticker_news(tickers, ticker_name)
    assert isinstance(news, list)
    assert all(isinstance(item, dict) for item in news)

def test_create_news_objects():
    ticker_news = [
        {
            "title": "News 1",
            "link": "https://example.com/news1",
            "thumbnail": {
                "resolutions": [{"url": "https://example.com/image1.jpg"}]
            },
            "providerPublishTime": "1631234567",
            "publisher": "Publisher 1"
        },
        {
            "title": "News 2",
            "link": "https://example.com/news2",
            "thumbnail": {
                "resolutions": [{"url": "https://example.com/image2.jpg"}]
            },
            "providerPublishTime": "1632345678",
            "publisher": "Publisher 2"
        }
    ]
    expected_result = [
        News(
            news_title="News 1",
            news_url="https://example.com/news1",
            news_image_url="https://example.com/image1.jpg",
            update_date="2021-09-10",
            publisher="Publisher 1"
        ),
        News(
            news_title="News 2",
            news_url="https://example.com/news2",
            news_image_url="https://example.com/image2.jpg",
            update_date="2021-09-23",
            publisher="Publisher 2"
        )
    ]
    assert create_news_objects(ticker_news) == expected_result

def test_get_news_yahoo():
    news = get_news_yahoo()
    assert isinstance(news, list)
    assert all(isinstance(item, list) for item in news)
    assert all(isinstance(news_item, News) for sublist in news for news_item in sublist)

def test_get_news():
    response = client.get("/api/v1/get_news")
    assert response.status_code == 200
    assert isinstance(response.json(), ListOfNews)
