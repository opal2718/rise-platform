# backend/main.py
import os
from fastapi import FastAPI, Request
import httpx
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import requests
import pickle
import pandas as pd
import numpy as np 

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

range = "5d"
period = "1h"
# GCS 마운트 경로
GCS_PATH = "/mnt/gcs"
FILENAME = "model.pkl"
FILEPATH = os.path.join(GCS_PATH, FILENAME)

@app.post("/predict")
async def predict(data: dict, request: Request):
    stock = data["text"]
    my_url = str(request.base_url).rstrip("/")
    data_url = f"{my_url}/data?stock={stock}&range={range}&period={period}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(data_url)
        request_data = resp.json()

    data_trend = request_data['trend']
    data_news = request_data['news_stock']
    data_news_sector = request_data['news_sector']

    # 데이터 전처리 및 피클링
    # pandas DF로 만들어야 함
    processed_data = pd.DataFrame({
        "trend": data_trend,
        "news": data_news,
        "news_sector": data_news_sector
    })

    with open(FILEPATH, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict_one(processed_data)
    # 다음 날 결과 나오면 학습하기 위해 데이터 저장

    with open(FILEPATH, "wb") as f:
        pickle.dump(model, f)

    return {"prediction": prediction}

@app.post("/process")
async def process(data: dict):
    A = data
    B = A  # 추론
    print("A")
    
    print(data)
    return {"output": B}

@app.post("/hello")
async def hello():
    return {"message": f"Hello!"}

#temp
# Add this to your ai1's main.py
@app.get("/data")
async def get_data(stock: str, range: str, period: str):
    # Implement logic here to fetch stock trend, news, news_sector
    # using yfinance, or other APIs.
    # Example (very basic placeholder):
    ticker = yf.Ticker(stock)
    hist = ticker.history(period=range, interval=period)
    trend_data = hist['Close'].tolist() if not hist.empty else []
    
    # Placeholder for news data
    news_stock_data = ["stock news 1", "stock news 2"]
    news_sector_data = ["sector news 1", "sector news 2"]
    
    return {
        "trend": trend_data,
        "news_stock": news_stock_data,
        "news_sector": news_sector_data
    }