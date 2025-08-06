# backend/main.py
import sysconfig; print(sysconfig.get_paths()["purelib"])
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/price")
def get_stock_price(ticker: str):
    data = yf.download(ticker, period="5d")
    return{"tikcer":ticker, "price":"you"}
    """
    if data.empty:
        return {"error": "Invalid ticker"}
    latest_price = data["Close"][-1]
    return {"ticker": ticker, "price": round(float(latest_price), 2)}"""
