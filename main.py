import os
import logging
import nltk

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends
from dotenv import load_dotenv
from models import Ticker, Sentiment, SentimentAnalysis
from utils import get_ticker_news, get_aggregated_results, get_sentiment, verify_api_key, get_sentiment_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('NEWS_API_KEY')
API_FUNCTION = "NEWS_SENTIMENT"
API_LIMIT = int(os.getenv('NEWS_API_LIMIT', 50))

# Initialize NLTK data needed for sentiment analysis
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

app = FastAPI(
    title="News Sentiment Analysis API",
    description="An API for sentiment analysis of news articles fetched from the AlphaVantage News API. The API fetches the latest news articles for a given ticker and performs aggregated sentiment analysis on the articles.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "message": "News Sentiment Analysis API is running"}

@app.post("/sentiment", response_model=SentimentAnalysis, tags=["Sentiment Analysis"])
async def analyze_sentiment(ticker: Ticker, sentiment_analyzer=Depends(get_sentiment_analyzer)):
    """
    Analyze sentiment of news articles for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sentiment analysis results including individual article sentiments and overall prediction
    """
    verify_api_key()
    
    ticker_symbol = ticker.ticker.upper()
    logger.info(f"Processing sentiment analysis request for ticker: {ticker_symbol}")
    
    summaries = get_ticker_news(ticker_symbol, API_KEY)
    if not summaries or len(summaries) == 0:
        raise HTTPException(
            status_code=404, 
            detail=f"No news articles found for ticker {ticker_symbol}"
        )

    sentiment_results = []
    sentiments = []
    
    for summary in summaries:
        label, score = get_sentiment(sentiment_analyzer, summary)
        sentiment_results.append((label, score))
        sentiments.append(Sentiment(
            label=label,
            score=float(score),
            summary=summary
        ))
    
    final_predict = get_aggregated_results(sentiment_results)
    
    logger.info(f"Completed sentiment analysis for {ticker_symbol}. Final prediction: {final_predict}")
    
    return SentimentAnalysis(
        ticker=ticker_symbol,
        final_predict=final_predict,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)