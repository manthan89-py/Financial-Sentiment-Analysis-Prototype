# News Sentiment Analysis API

## Project Description

The News Sentiment Analysis API is a FastAPI-based application that analyzes sentiment from news articles related to specific stock tickers. The application fetches the latest news articles from AlphaVantage's News API for a given ticker symbol, performs NLP preprocessing on the article summaries, and conducts sentiment analysis using a pre-trained DistilBERT model. The API then aggregates these individual sentiment scores to provide an overall sentiment prediction (POSITIVE, NEGATIVE, or NEUTRAL) for the ticker.

Key features:
- Fetches real-time news data from AlphaVantage
- Uses NLTK for text preprocessing (stopword removal, lemmatization)
- Utilizes Hugging Face Transformers for sentiment analysis
- Provides both individual article sentiment scores and an aggregated prediction
- Includes robust error handling and logging
- Containerized for easy deployment

## Project Structure

```
├── main.py            # FastAPI application entry point
├── models.py          # Pydantic models for request/response validation
├── utils.py           # Utility functions for preprocessing and sentiment analysis
├── Dockerfile         # Container configuration
├── pyproject.toml     # Project metadata and dependencies (Poetry format)
├── README.md          # Project documentation
├── requirements.txt   # Dependencies list
└── uv.lock            # Lock file for uv package manager
```

## Assumptions and Design Choices

### Architecture Decisions
1. **Single Service Architecture**: We chose a monolithic approach for simplicity and ease of deployment, keeping all functionality in a single service.
2. **RESTful API Design**: Implemented a RESTful API pattern for clear endpoint structure and intuitive use.
3. **Containerization**: Docker was chosen for containerization to ensure consistent behavior across different environments.

### Technical Assumptions
1. **News Source**: We assumed AlphaVantage provides reliable and up-to-date financial news data suitable for sentiment analysis.
2. **API Limits**: The design accommodates rate limits imposed by AlphaVantage by implementing proper error handling for API failures.
3. **Language**: The application assumes news articles are in English since the sentiment analysis model is trained on English text.
4. **Sentiment Categories**: We simplified sentiment to three categories (POSITIVE, NEGATIVE, NEUTRAL) rather than a continuous scale.

### Implementation Choices
1. **Pre-trained Model**: DistilBERT was chosen for sentiment analysis due to its balance of accuracy and performance compared to larger models.
2. **Text Preprocessing**: Implemented stopword removal and lemmatization to improve sentiment analysis accuracy by focusing on meaningful content words.
3. **Caching Strategy**: Used LRU caching for the sentiment analysis model to minimize resource usage while maintaining performance.
4. **Error Handling**: Implemented comprehensive error handling with appropriate HTTP status codes and informative error messages.
5. **Aggregation Method**: Sentiment aggregation uses a simple majority vote rather than weighted averaging to prioritize clear signals.

### Performance Considerations
1. **Model Loading**: The sentiment analysis model is loaded once at startup and cached to avoid repeated loading overhead.
2. **Dependency Injection**: Used FastAPI's dependency injection to ensure efficient resource utilization.
3. **Concurrency**: The application leverages FastAPI's asynchronous capabilities for handling concurrent requests efficiently.

### Potential Limitations
1. **Domain Specificity**: The general-purpose sentiment model may not capture financial domain-specific sentiment nuances.
2. **Real-time Processing**: Large volumes of news articles might cause processing delays, especially with the transformer-based model.
3. **External API Dependency**: The application's functionality depends on the availability and reliability of the AlphaVantage API.

## Useful Functions in the Code

1. **`preprocess_summary()`**: Cleans text data by removing stopwords and applying lemmatization
   ```python
   def preprocess_summary(summary: str) -> str:
       """Preprocess text by removing stopwords and lemmatizing."""
   ```

2. **`get_ticker_news()`**: Fetches news articles for a specific ticker from AlphaVantage
   ```python
   def get_ticker_news(ticker: str, api_key: str, function: str, limit: int) -> Optional[List[str]]:
       """Fetch news articles for a specific ticker from AlphaVantage API."""
   ```

3. **`get_sentiment()`**: Analyzes sentiment of text using the DistilBERT model
   ```python
   def get_sentiment(sentiment_analyzer, text: str) -> tuple:
       """Analyze sentiment of a text."""
   ```

4. **`get_aggregated_results()`**: Aggregates individual sentiment scores to determine overall sentiment
   ```python
   def get_aggregated_results(sentiment_results: List[tuple]) -> str:
       """Aggregate sentiment results to determine overall sentiment."""
   ```

5. **`get_sentiment_analyzer()`**: Initializes and caches the sentiment analysis model
   ```python
   @lru_cache(maxsize=1)
   def get_sentiment_analyzer():
       """Initialize and cache the sentiment analysis model."""
   ```

## How to Run Project Locally

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, if using uv:
   ```bash
   uv pip install -r requirements.txt
   ```
4. Create a `.env` file with your AlphaVantage API key:
   ```
   NEWS_API_KEY=your_api_key_here
   ```
5. Run the FastAPI application:
   ```bash
   uvicorn main:app --reload
   ```
6. Access the API documentation at http://localhost:8000/docs

## Response Description

The API returns a JSON response with the following structure:

```json
{
  "ticker": "AAPL",              // The ticker symbol that was analyzed
  "final_predict": "POSITIVE",   // Aggregated sentiment prediction
}
```

## How to Build the Project

Build the Docker image:

```bash
docker build -t webapp:sentiment-analysis .
```

This creates a containerized version of the application with all dependencies included.

## How to Run the Docker Build

Run the Docker container:

```bash
docker run -p 8000:8000 --env-file .env webapp:sentiment-analysis
```

The API will be accessible at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## API Endpoints

- `GET /`: Health check endpoint
- `POST /sentiment`: Analyze sentiment for a given ticker
  - Request body: `{"ticker": "AAPL"}`
  - Returns: Sentiment analysis results including article sentiments and overall prediction

## Dependencies

- FastAPI: Web framework
- Pydantic: Data validation
- Transformers: Pre-trained NLP models
- NLTK: Natural language processing
- Requests: HTTP client
- Python-dotenv: Environment variable management
- Uvicorn: ASGI server