'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
} from 'chart.js';
import { ChartOptions } from 'chart.js';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

interface StockChartData {
  labels: string[];
  datasets: {
    label: string;
    data: (number | null)[];
    borderColor: string;
    backgroundColor: string;
    fill: boolean;
    tension: number;
    pointRadius?: number;
    pointHoverRadius?: number;
    borderDash?: number[];
  }[];
}

interface TrendingStockData {
  ticker: string;
  chartData: StockChartData | null;
  currentPrice: number | null;
  predictedValue: number | null;
  recommendation: '추천' | '비추천' | '중립' | null;
  error: string;
}

export default function PredictPage() {
  const [ticker, setTicker] = useState('');
  const [currentPrice, setCurrentPrice] = useState<null | number>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recommendation, setRecommendation] = useState<'추천' | '비추천' | '중립' | null>(null);
  const [chartData, setChartData] = useState<StockChartData | null>(null);
  const [predictedValue, setPredictedValue] = useState<null | number>(null);

  const trendingTickers = ["AAPL", "TSLA", "NVDA"];
  const [trendingStocksData, setTrendingStocksData] = useState<TrendingStockData[]>([]);

  const myListTickers = ["GOOG", "MSFT"];
  const [myListStocksData, setMyListStocksData] = useState<TrendingStockData[]>([]);


  const fetchStockData = async (stockTicker: string, isTrending: boolean = false) => {
    // 'data' 변수를 'const'로 변경
    const data: TrendingStockData = {
      ticker: stockTicker,
      chartData: null,
      currentPrice: null,
      predictedValue: null,
      recommendation: null,
      error: '',
    };

    try {
      // 1. Fetch 90 days of historical data
      const historicalRes = await fetch(`http://34.16.110.5:8001/data?stock=${stockTicker}&period=1d`);
      if (!historicalRes.ok) {
        throw new Error(`과거 데이터 API 요청 실패: ${historicalRes.status} ${historicalRes.statusText}`);
      }
      const historicalData = await historicalRes.json();

      if (historicalData.error || !historicalData.processed_features_for_prediction) {
        throw new Error(historicalData.error || '과거 데이터를 가져오지 못했습니다.');
      }

      const featuresForPrediction = historicalData.processed_features_for_prediction;
      const latestClosePrice = featuresForPrediction.Close;
      data.currentPrice = latestClosePrice;

      // 2. Fetch 1-day future prediction
      const predictionRes = await fetch('http://34.16.110.5:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: stockTicker }),
      });
      if (!predictionRes.ok) {
        throw new Error(`예측 API 요청 실패: ${predictionRes.status} ${predictionRes.statusText}`);
      }
      const predictionResult = await predictionRes.json();

      if (predictionResult.error || typeof predictionResult.prediction === 'undefined') {
        throw new Error(predictionResult.error || '예측값을 가져오지 못했습니다.');
      }
      data.predictedValue = predictionResult.prediction;

      // Determine recommendation
      const change = predictionResult.prediction - (latestClosePrice || 0);
      if (latestClosePrice && change > 0.01 * latestClosePrice) {
        data.recommendation = '추천';
      } else if (latestClosePrice && change < -0.01 * latestClosePrice) {
        data.recommendation = '비추천';
      } else {
        data.recommendation = '중립';
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

      data.chartData = {
        labels: fullChartLabels,
        datasets: [
          {
            label: '과거 주가',
            data: historicalChartDataPoints,
            borderColor: '#6366f1',
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.3,
            pointRadius: isTrending ? 0 : 3, // Smaller points for trending charts
            pointHoverRadius: isTrending ? 0 : 6,
          },
          {
            label: '예측 주가',
            data: predictionChartDataPoints,
            borderColor: '#f97316',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            fill: false,
            tension: 0.3,
            pointRadius: isTrending ? 0 : 5,
            pointHoverRadius: isTrending ? 0 : 8,
          }
        ],
      };

    } catch (err: unknown) {
      console.error(`Error fetching data for ${stockTicker}:`, err);
      data.error = 'API 요청 실패: ' + (err instanceof Error ? err.message : String(err));
      data.chartData = {
        labels: ["Error"],
        datasets: [
            {
                label: '데이터 없음',
                data: [0],
                borderColor: '#ef4444',
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.3,
            }
        ]
      }
    }
    return data;
  };

  const fetchMainPrediction = async () => {
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

    const result = await fetchStockData(ticker);
    setCurrentPrice(result.currentPrice);
    setPredictedValue(result.predictedValue);
    setRecommendation(result.recommendation);
    setChartData(result.chartData);
    setError(result.error);
    setLoading(false);
  };

  useEffect(() => {
    const fetchTrendingAndMyListData = async () => {
      // Fetch trending stocks
      const trendingResults = await Promise.all(
        trendingTickers.map(t => fetchStockData(t, true))
      );
      setTrendingStocksData(trendingResults);

      // Fetch My List stocks
      const myListResults = await Promise.all(
        myListTickers.map(t => fetchStockData(t, true))
      );
      setMyListStocksData(myListResults);
    };

    fetchTrendingAndMyListData();
  }, [trendingTickers, myListTickers]); // 의존성 배열에 추가

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
            color: '#374151', // Darker gray for legend labels
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#6366f1',
        borderWidth: 1,
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
          maxTicksLimit: 10,
          color: '#6b7280', // Gray for x-axis ticks
        },
        grid: {
            color: '#e5e7eb', // Light gray grid lines
        }
      },
      y: {
        beginAtZero: false,
        ticks: {
            color: '#6b7280', // Gray for y-axis ticks
        },
        grid: {
            color: '#e5e7eb', // Light gray grid lines
        }
      }
    }
  };

  // Options for smaller charts in trending/my list sections
  const smallChartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false } // Disable tooltips for small charts
    },
    scales: {
      x: { display: false },
      y: { display: false }
    },
    elements: {
        point: {
            radius: 0, // Hide points
        },
        line: {
            tension: 0.4, // Smooth lines
            borderWidth: 2,
        }
    },
  };

  return (
    <main className="flex flex-col min-h-screen bg-[#f5f7f9]">
      <Header />
      <section className="max-w-5xl mx-auto px-6 py-20 w-full">
        <h1 className="text-4xl font-semibold text-gray-700 mb-6 text-center">주가 예측</h1>
        <p className="text-gray-500 mb-12 text-center">AI를 활용하여 선택한 종목의 향후 주가를 예측합니다.</p>

        {/* Search Section */}
        <div className="mb-12 bg-white p-8 rounded-lg shadow-lg">
          <label htmlFor="stock-search" className="block text-lg font-medium text-gray-700 mb-3">종목 검색</label>
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <input
              id="stock-search"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="예: AAPL, 삼성전자"
              className="flex-1 px-4 py-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-800 text-lg"
            />
            <button
              onClick={fetchMainPrediction}
              className="px-8 py-3 bg-indigo-600 text-white rounded-md shadow-md hover:bg-indigo-700 transition duration-300 ease-in-out font-semibold text-lg"
              disabled={loading || !ticker}
            >
              {loading ? '예측 중...' : '예측하기'}
            </button>
          </div>
          {currentPrice !== null && predictedValue !== null && (
            <div className="mt-6 p-5 bg-indigo-50 rounded-lg border border-indigo-200">
                <p className="text-xl text-gray-800 font-medium">📈 {ticker.toUpperCase()}의 최신 종가는 <strong className="text-indigo-700">{currentPrice?.toFixed(2)}</strong>입니다.</p>
                <p className="text-xl text-gray-800 mt-3 font-medium">🔮 1일 후 예측 주가는 <strong className="text-orange-600">{predictedValue?.toFixed(2)}</strong>입니다.</p>
                {recommendation && (
                    <p className={`mt-4 text-xl font-bold text-white px-5 py-2 inline-block rounded-full shadow-md ${
                        recommendation === '추천' ? 'bg-green-500' :
                        recommendation === '비추천' ? 'bg-red-500' :
                        'bg-gray-500'
                    }`}>
                        예측에 따른 추천 결과: <span className="text-white">{recommendation}</span>
                    </p>
                )}
            </div>
          )}

          {chartData && (
            <div className="my-8 bg-white p-6 rounded-lg shadow-md border border-gray-200">
              <h3 className="text-xl font-semibold text-gray-700 mb-4">주가 추이 및 예측</h3>
              <Line data={chartData} options={chartOptions} />
            </div>
          )}


          {error && (
            <p className="mt-4 text-red-600 bg-red-100 p-3 rounded-md border border-red-200">⚠️ {error}</p>
          )}
        </div>

        {/* Trending Section */}
        <div className="mb-12">
          <h2 className="text-3xl font-semibold text-gray-700 mb-6">🔥 트렌딩 종목</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {trendingStocksData.map((stock, index) => (
              <div key={index} className="bg-white p-5 rounded-xl shadow-lg border border-gray-100 flex flex-col items-center justify-between">
                <h3 className="text-xl text-center font-semibold text-gray-800 mb-3">{stock.ticker}</h3>
                {stock.chartData ? (
                  <div className="w-full h-32 mb-3">
                    <Line data={stock.chartData} options={smallChartOptions} />
                  </div>
                ) : (
                  <div className="w-full h-32 flex items-center justify-center text-gray-500 text-sm">
                    {stock.error || '데이터 로딩 중...'}
                  </div>
                )}
                {stock.predictedValue !== null && stock.currentPrice !== null && (
                    <div className="text-center text-sm">
                        <p className="text-gray-600">오늘 종가: <span className="font-semibold">{stock.currentPrice.toFixed(2)}</span></p>
                        <p className="text-gray-600">예측 종가: <span className="font-semibold text-orange-500">{stock.predictedValue.toFixed(2)}</span></p>
                    </div>
                )}
                {stock.recommendation && (
                    <span className={`mt-3 px-3 py-1 text-xs font-bold rounded-full text-white ${
                        stock.recommendation === '추천' ? 'bg-green-400' :
                        stock.recommendation === '비추천' ? 'bg-red-400' :
                        'bg-gray-400'
                    }`}>
                        {stock.recommendation}
                    </span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* My List Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-semibold text-gray-700 mb-6">📌 내 관심 종목</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {myListStocksData.map((stock, index) => (
              <div key={index} className="bg-white p-5 rounded-xl shadow-lg border border-gray-100 flex flex-col items-center justify-between">
                <h3 className="text-xl text-center font-semibold text-gray-800 mb-3">{stock.ticker}</h3>
                {stock.chartData ? (
                  <div className="w-full h-32 mb-3">
                    <Line data={stock.chartData} options={smallChartOptions} />
                  </div>
                ) : (
                  <div className="w-full h-32 flex items-center justify-center text-gray-500 text-sm">
                    {stock.error || '데이터 로딩 중...'}
                  </div>
                )}
                {stock.predictedValue !== null && stock.currentPrice !== null && (
                    <div className="text-center text-sm">
                        <p className="text-gray-600">오늘 종가: <span className="font-semibold">{stock.currentPrice.toFixed(2)}</span></p>
                        <p className="text-gray-600">예측 종가: <span className="font-semibold text-orange-500">{stock.predictedValue.toFixed(2)}</span></p>
                    </div>
                )}
                {stock.recommendation && (
                    <span className={`mt-3 px-3 py-1 text-xs font-bold rounded-full text-white ${
                        stock.recommendation === '추천' ? 'bg-green-400' :
                        stock.recommendation === '비추천' ? 'bg-red-400' :
                        'bg-gray-400'
                    }`}>
                        {stock.recommendation}
                    </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}