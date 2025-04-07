import os
import requests
from dotenv import load_dotenv
from typing import List, Optional
from fastapi import HTTPException

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functools import lru_cache

from transformers import pipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv('NEWS_API_KEY')
API_FUNCTION = "NEWS_SENTIMENT"
API_LIMIT = int(os.getenv('NEWS_API_LIMIT', 50))


@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    """
    Initialize and cache the sentiment analysis model.
    """
    logger.info("Initializing sentiment analysis model")
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

def get_ticker_news(ticker: str, api_key: str, function: str = API_FUNCTION, limit: int = API_LIMIT) -> Optional[List[str]]:
    """
    Fetch news articles for a specific ticker from AlphaVantage API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: API key for AlphaVantage
        function: API function to call
        limit: Maximum number of articles to fetch
        
    Returns:
        List of preprocessed article summaries or None if request fails
    """
    url = f'https://www.alphavantage.co/query?function={function}&tickers={ticker}&limit={limit}&sort=LATEST&apikey={api_key}'
    try:
        summaries = []
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'feed' not in data:
            logger.warning(f"No feed in response for ticker {ticker}: {data}")
            return None
            
        for item in data['feed']:
            if 'summary' in item:
                summary = preprocess_summary(item['summary'])
                summaries.append(summary)
        
        logger.info(f"Retrieved {len(summaries)} news articles for ticker {ticker}")
        return summaries
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return None
    
def preprocess_summary(summary: str) -> str:
    """
    Preprocess text by removing stopwords and lemmatizing.
    
    Args:
        summary: The text to preprocess
        
    Returns:
        Preprocessed text
    """
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(summary)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        processed_summary = ' '.join(tokens)
        return processed_summary
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return summary  # Return original if preprocessing fails

def get_aggregated_results(sentiment_results: List[tuple]) -> str:
    """
    Aggregate sentiment results to determine overall sentiment.
    
    Args:
        sentiment_results: List of (sentiment, score) tuples
        
    Returns:
        Final sentiment prediction
    """
    if not sentiment_results:
        return "NEUTRAL"
        
    pos = 0
    neg = 0
    neutral = 0
    for sentiment, _ in sentiment_results:
        if sentiment == 'POSITIVE':
            pos += 1
        elif sentiment == 'NEGATIVE':
            neg += 1
        else:
            neutral += 1
    max_value = max(pos, neg, neutral)
    if max_value == pos:
        return 'POSITIVE'
    elif max_value == neg:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'  # Changed from POSITIVE to NEUTRAL for equal cases
    
def get_sentiment(sentiment_analyzer, text: str) -> tuple:
    """
    Analyze sentiment of a text.
    
    Args:
        sentiment_analyzer: The sentiment analysis model
        text: The text to analyze
        
    Returns:
        Tuple of (sentiment label, confidence score)
    """
    try:
        result = sentiment_analyzer(text)[0]
        return (result['label'], result['score'])
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return ("NEUTRAL", 0.5)  # Fallback value
    
def verify_api_key():
    """
    Verify that the API key is configured.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="API key not configured. Please set the NEWS_API_KEY environment variable."
        )
