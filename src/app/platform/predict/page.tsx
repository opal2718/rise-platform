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
} from 'chart.js';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip);

export default function PredictPage() {
  const [ticker, setTicker] = useState('');
  const [price, setPrice] = useState<null | number>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recommendation, setRecommendation] = useState<'추천' | '비추천' | '중립' | null>(null);

  const fetchPrice = async () => {
    setLoading(true);
    setError('');
    setPrice(null);
    setRecommendation(null);
    try {
      const res = await fetch(`http://localhost:8000/api/price?ticker=${ticker}`);
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setPrice(data.price);
        const change = data.price - 100; // 임시 기준값 100
        if (change > 5) setRecommendation('추천');
        else if (change < -5) setRecommendation('비추천');
        else setRecommendation('중립');
      }
    } catch (err) {
      setError('API 요청 실패'+err);
    } finally {
      setLoading(false);
    }
  };

  const generateMockHourlyLabels = () => {
    const days = ['월', '화', '수', '목', '금', '토', '일'];
    const hours = ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00'];
    return days.flatMap(day => hours.map(hour => `${day} ${hour}`)).concat(['예측1', '예측2', '예측3', '예측4', '예측5']);
  };

  const allLabels = generateMockHourlyLabels();
  const historicalData = Array.from({ length: 49 }, (_, i) => 100 + Math.sin(i / 3) * 5 + Math.random() * 2);
  const predictedData = [110, 111, 112, 113, 114];

  const mockChartData = {
    labels: allLabels,
    datasets: [
      {
        label: '1시간 단위 주가 (7일)',
        data: [...historicalData, ...Array(5).fill(null)], // 길이 54
        borderColor: '#6366f1',
        fill: false,
        tension: 0.3,
      },
      {
        label: '예측 주가',
        data: [...Array(49).fill(null), ...predictedData], // 예측만 있는 구간
        borderColor: '#f97316',
        borderDash: [5, 5],
        fill: false,
        tension: 0.3,
      }
    ],
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
              placeholder="예: AAPL, 삼성전자, TSLA"
              className="flex-1 px-4 py-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 text-blue-600"
            />
            <button
              onClick={fetchPrice}
              className="px-6 py-3 bg-indigo-600 text-white rounded-md shadow hover:bg-indigo-700 transition"
              disabled={loading || !ticker}
            >
              {loading ? '예측 중...' : '예측하기'}
            </button>
          </div>
          {price !== null && (
            <>
              <p className="mt-4 text-lg text-gray-700">📈 {ticker.toUpperCase()}의 최신 종가는 <strong>{price}</strong>입니다.</p>
              <div className="my-6">
                <Line data={mockChartData} options={{ responsive: true, plugins: { legend: { display: false } } }} />
              </div>
            </>
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
                <Line data={mockChartData} options={{ responsive: true, plugins: { legend: { display: false } } }} />
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
                <Line data={mockChartData} options={{ responsive: true, plugins: { legend: { display: false } } }} />
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
