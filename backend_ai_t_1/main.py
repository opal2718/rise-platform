# backend/main.py
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

@app.post("/process")
async def process(data: dict):
    A = data["input"]
    B = A  # 추론
    print("A")
    
    print(data)
    return {"output": B}
