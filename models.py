from pydantic import BaseModel

class Ticker(BaseModel):
    ticker: str
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL"
            }
        }

class Sentiment(BaseModel):
    label: str
    score: float
    summary: str

class SentimentAnalysis(BaseModel):
    ticker: str
    final_predict: str