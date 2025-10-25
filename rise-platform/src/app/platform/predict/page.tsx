'use client';

import { useState } from 'react';
import Header from '@/components/Header';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend // Import Legend
} from 'chart.js';
import { ChartOptions } from 'chart.js'; // Removed ChartData as it's not directly used for state, StockChartData is sufficient

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

// Define a type for your chart data structure
interface StockChartData {
  labels: string[];
  datasets: {
    label: string;
    data: (number | null)[];
    borderColor: string;
    backgroundColor: string; // Added for consistency
    fill: boolean;
    tension: number;
    pointRadius?: number;
    pointHoverRadius?: number;
    borderDash?: number[];
  }[];
}


export default function PredictPage() {
  const [ticker, setTicker] = useState('');
  const [currentPrice, setCurrentPrice] = useState<null | number>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recommendation, setRecommendation] = useState<'추천' | '비추천' | '중립' | null>(null);
  const [chartData, setChartData] = useState<StockChartData | null>(null);
  const [predictedValue, setPredictedValue] = useState<null | number>(null);

  const fetchPredictionAndHistoricalData = async () => {
    setLoading(true);
    setError('');
    setCurrentPrice(null);
    setRecommendation(null);
    setChartData(null);
    setPredictedValue(null);

    if (!ticker) {
      setError('종목을 입력해주세요.');
      setLoading(false);
      return;
    }

    try {
      // 1. Fetch 90 days of historical data from ai2:8001/data
      const historicalRes = await fetch(`http://34.16.110.5:8001/data?stock=${ticker}&period=1d`);
      // Check if response is okay before parsing JSON
      if (!historicalRes.ok) {
        throw new Error(`과거 데이터 API 요청 실패: ${historicalRes.status} ${historicalRes.statusText}`);
      }
      const historicalData = await historicalRes.json();

      if (historicalData.error || !historicalData.processed_features_for_prediction) {
        throw new Error(historicalData.error || '과거 데이터를 가져오지 못했습니다.');
      }

      const featuresForPrediction = historicalData.processed_features_for_prediction;
      const latestClosePrice = featuresForPrediction.Close;
      setCurrentPrice(latestClosePrice);

      // 2. Fetch 1-day future prediction from ai1:8000/predict
      const predictionRes = await fetch('http://34.16.110.5:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: ticker }),
      });
      // Check if response is okay before parsing JSON
      if (!predictionRes.ok) {
        throw new Error(`예측 API 요청 실패: ${predictionRes.status} ${predictionRes.statusText}`);
      }
      const predictionResult = await predictionRes.json();

      if (predictionResult.error || typeof predictionResult.prediction === 'undefined') {
        throw new Error(predictionResult.error || '예측값을 가져오지 못했습니다.');
      }
      setPredictedValue(predictionResult.prediction);

      // Determine recommendation
      const change = predictionResult.prediction - (latestClosePrice || 0); // Use 0 if latestClosePrice is null/undefined
      if (latestClosePrice && change > 0.01 * latestClosePrice) {
        setRecommendation('추천');
      } else if (latestClosePrice && change < -0.01 * latestClosePrice) {
        setRecommendation('비추천');
      } else {
        setRecommendation('중립');
      }

      // Prepare chart data
      const historicalClosePrices: number[] = [];
      const historicalLabels: string[] = [];

      if (latestClosePrice !== null && typeof latestClosePrice !== 'undefined') {
        historicalClosePrices.push(latestClosePrice);
        historicalLabels.push('오늘');
      }

      for (let i = 1; i <= 90; i++) {
        const laggedClose = featuresForPrediction[`Close_lag_${i}`];
        if (laggedClose !== null && typeof laggedClose !== 'undefined') {
          historicalClosePrices.unshift(laggedClose);
          historicalLabels.unshift(`D-${i}`);
        } else {
            break;
        }
      }

      const fullChartLabels = [...historicalLabels, '예측'];
      const historicalChartDataPoints = [...historicalClosePrices, null];
      const predictionChartDataPoints = [...Array(historicalClosePrices.length).fill(null), predictionResult.prediction];


      setChartData({
        labels: fullChartLabels,
        datasets: [
          {
            label: '과거 주가',
            data: historicalChartDataPoints,
            borderColor: '#6366f1',
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 6,
          },
          {
            label: '예측 주가',
            data: predictionChartDataPoints,
            borderColor: '#f97316',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            fill: false,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
          }
        ],
      });

    } catch (err: unknown) { // Use 'unknown' here
      console.error("Error during fetch:", err);
      // Type-check 'err' to extract the message safely
      setError('API 요청 실패: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  };

  const generateMockChartData = (stockName: string): StockChartData => {
    const mockLabels = Array.from({ length: 7 }, (_, i) => `Day ${i + 1}`);
    const mockData = Array.from({ length: 7 }, (_, i) => 100 + Math.sin(i) * 10 + Math.random() * 5);
    return {
      labels: mockLabels,
      datasets: [
        {
          label: `${stockName} 주가`,
          data: mockData,
          borderColor: '#a78bfa',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          pointRadius: 0,
        },
      ],
    };
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      }
    },
    hover: {
      mode: 'nearest',
      intersect: true
    },
    scales: {
      x: {
        ticks: {
          autoSkip: true,
          maxTicksLimit: 10
        }
      },
      y: {
        beginAtZero: false
      }
    }
  };


  return (
    <main className="flex flex-col min-h-screen bg-[#f5f7f9]">
      <Header />
      <section className="max-w-5xl mx-auto px-6 py-20">
        <h1 className="text-4xl font-semibold text-gray-700 mb-6 text-center">주가 예측</h1>
        <p className="text-gray-500 mb-12 text-center">AI를 활용하여 선택한 종목의 향후 주가를 예측합니다.</p>

        {/* Search Section */}
        <div className="mb-12">
          <label htmlFor="stock-search" className="block text-sm font-medium text-gray-600 mb-2">종목 검색</label>
          <div className="flex gap-4">
            <input
              id="stock-search"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="예: AAPL, 삼성전자"
              className="flex-1 px-4 py-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 text-blue-600"
            />
            <button
              onClick={fetchPredictionAndHistoricalData}
              className="px-6 py-3 bg-indigo-600 text-white rounded-md shadow hover:bg-indigo-700 transition"
              disabled={loading || !ticker}
            >
              {loading ? '예측 중...' : '예측하기'}
            </button>
          </div>
          {currentPrice !== null && predictedValue !== null && (
            <div className="mt-4 p-4 bg-white rounded-lg shadow-md">
                <p className="text-lg text-gray-700">📈 {ticker.toUpperCase()}의 최신 종가는 <strong>{currentPrice?.toFixed(2)}</strong>입니다.</p>
                <p className="text-lg text-gray-700 mt-2">🔮 1일 후 예측 주가는 <strong>{predictedValue?.toFixed(2)}</strong>입니다.</p>
            </div>
          )}

          {chartData && (
            <div className="my-6 bg-white p-4 rounded-lg shadow-md">
              <Line data={chartData} options={chartOptions} />
            </div>
          )}

          {recommendation && (
            <p className="mt-2 text-base font-semibold text-center text-white px-4 py-2 inline-block rounded-full shadow bg-gradient-to-r from-green-400 via-yellow-300 to-red-400">
              예측에 따른 추천 결과: <span className="text-black">{recommendation}</span>
            </p>
          )}
          {error && (
            <p className="mt-4 text-red-500">⚠️ {error}</p>
          )}
        </div>

        {/* Trending Section */}
        <div className="mb-12">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">🔥 트렌딩 종목</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {["삼성전자", "AAPL", "TSLA", "NVDA"].map((stock, index) => (
              <div key={index} className="bg-white p-4 rounded-lg shadow">
                <h3 className="text-center font-semibold text-gray-700 mb-2">{stock}</h3>
                <Line data={generateMockChartData(stock)} options={{ responsive: true, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } }} />
              </div>
            ))}
          </div>
        </div>

        {/* My List Section */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">📌 내 관심 종목</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {["GOOG", "MSFT"].map((stock, index) => (
              <div key={index} className="bg-white p-4 rounded-lg shadow">
                <h3 className="text-center font-semibold text-gray-700 mb-2">{stock}</h3>
                <Line data={generateMockChartData(stock)} options={{ responsive: true, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } }} />
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}