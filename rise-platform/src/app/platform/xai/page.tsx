'use client';

import { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Line } from "react-chartjs-2";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import Header from '@/components/Header';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineController,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { ChartOptions } from 'chart.js';
import { useMemo } from 'react'; // useMemo import 추가

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineController,
  Title,
  Tooltip,
  Legend
);

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

interface InfluenceFactor {
    factor: string;
    description: string;
    direction: '상승' | '하락' | '중립';
    strength: number;
}

export default function PredictExplainPage() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentPrice, setCurrentPrice] = useState<null | number>(null);
  const [predictedValue, setPredictedValue] = useState<null | number>(null);
  const [recommendation, setRecommendation] = useState<'추천' | '비추천' | '중립' | null>(null);
  const [chartData, setChartData] = useState<StockChartData | null>(null);
  const [influenceFactors, setInfluenceFactors] = useState<InfluenceFactor[]>([]);

  // useMemo를 사용하여 featureNames 배열이 한 번만 생성되도록 함
  const featureNames = useMemo(() => [
    "Open", "High", "Low", "Close", "Adj Close", "Volume", // Base features
    ...Array.from({ length: 90 }, (_, i) => `Close_lag_${i + 1}`),
    ...Array.from({ length: 90 }, (_, i) => `Volume_lag_${i + 1}`),
    "RSI_14", "MACD", "MACD_Signal", "Bollinger_Upper", "Bollinger_Lower",
    "MA_5", "MA_10", "MA_20", "MA_60", "MA_120",
    // 재무제표 관련 피처 (예시)
    "법인세차감전 순이익OFS", "법인세차감전 순이익CFS", "당기순이익(손실)OFS", "당기순이익(손실)CFS",
    "stock_news_total_count", "sector_sentiment_avg", "sector_relevance_avg", "stock_sentiment_avg",
    "총포괄손익OFS", "총포괄손익CFS",
    // 기타 필요한 피처들을 여기에 추가해주세요 (백엔드와 순서 일치 필수)
  ], []);


  const fetchStockAndExplanationData = async () => {
    setLoading(true);
    setError('');
    setCurrentPrice(null);
    setPredictedValue(null);
    setRecommendation(null);
    setChartData(null);
    setInfluenceFactors([]);

    if (!ticker) {
      setError('종목을 입력해주세요.');
      setLoading(false);
      return;
    }

    try {
      // 1. Fetch 90 days of historical data
      const historicalRes = await fetch(`http://34.16.110.5:8001/data?stock=${ticker}&period=1d`);
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

      // 2. Fetch 1-day future prediction
      const predictionRes = await fetch('http://34.16.110.5:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: ticker }),
      });
      if (!predictionRes.ok) {
        throw new Error(`예측 API 요청 실패: ${predictionRes.status} ${predictionRes.statusText}`);
      }
      const predictionResult = await predictionRes.json();

      if (predictionResult.error || typeof predictionResult.prediction === 'undefined') {
        throw new Error(predictionResult.error || '예측값을 가져오지 못했습니다.');
      }
      setPredictedValue(predictionResult.prediction);

      // Determine recommendation
      const change = predictionResult.prediction - (latestClosePrice || 0);
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

      // 3. Fetch explanation data
      const explainRes = await fetch('http://34.16.110.5:8002/explain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: ticker }),
      });
      if (!explainRes.ok) {
        throw new Error(`설명 API 요청 실패: ${explainRes.status} ${explainRes.statusText}`);
      }
      const explainData = await explainRes.json();

      const topNToShow = 5; // 화면에 표시할 최대 영향 요인 개수
      const significantThreshold = 0.01; // 유의미하다고 판단할 SHAP 값의 최소 임계치 (조정 가능)

      if (Array.isArray(explainData) && explainData.length > 0) {
          const factorContributions = featureNames
              .map((name, index) => ({
                  factor: name,
                  contribution: explainData[index] || 0,
              }));

          // 유의미한 SHAP 값을 가진 요인들
          const significantFactors = factorContributions
              .filter(fc => Math.abs(fc.contribution) > significantThreshold)
              .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

          let factorsToDisplay = [];

          if (significantFactors.length < topNToShow) {
              // 유의미한 요인이 topNToShow 개수보다 적으면,
              // 모든 요인을 절대값 기준으로 정렬하여 상위 topNToShow 개를 표시
              factorsToDisplay = factorContributions
                  .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                  .slice(0, topNToShow);
          } else {
              // 유의미한 요인이 충분하면, 그 중에서 상위 topNToShow 개를 표시
              factorsToDisplay = significantFactors.slice(0, topNToShow);
          }

          const processedInfluenceFactors: InfluenceFactor[] = factorsToDisplay.map(fc => {
              const direction = fc.contribution > 0 ? '상승' : fc.contribution < 0 ? '하락' : '중립';
              // SHAP 값의 절대값에 비례하여 강도 설정. 0인 경우 최소 강도 1로 설정.
              const strength = Math.max(1, Math.min(3, Math.ceil(Math.abs(fc.contribution) * 100))); // 스케일링 인자 조정 가능
              return {
                  factor: fc.factor.replace(/_/g, ' '),
                  description: `${fc.factor.replace(/_/g, ' ')}의 기여도: ${fc.contribution.toFixed(4)}`,
                  direction,
                  strength,
              };
          });
          setInfluenceFactors(processedInfluenceFactors);
      } else {
        setError("설명 데이터를 가져오지 못했습니다.");
      }

    } catch (err: unknown) {
      console.error(`Error fetching data for ${ticker}:`, err);
      setError('데이터를 가져오는 데 실패했습니다: ' + (err instanceof Error ? err.message : String(err)));
      setChartData({
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
      })
    } finally {
      setLoading(false);
    }
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
            color: '#374151',
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
          color: '#6b7280',
        },
        grid: {
            color: '#e5e7eb',
        }
      },
      y: {
        beginAtZero: false,
        ticks: {
            color: '#6b7280',
        },
        grid: {
            color: '#e5e7eb',
        }
      }
    }
  };

  return (
    <main className="flex flex-col min-h-screen bg-[#f5f7f9]">
      <Header />
      <section className="max-w-5xl mx-auto px-6 py-20 w-full">
        <h1 className="text-4xl font-semibold text-gray-700 mb-6 text-center">AI 예측 설명 (XAI)</h1>
        <p className="text-gray-500 mb-12 text-center">AI가 주가를 예측한 근거와 주요 영향 요인을 분석합니다.</p>

        {/* Search Section */}
        <div className="mb-12 bg-white p-8 rounded-lg shadow-lg">
          <label htmlFor="stock-search" className="block text-lg font-medium text-gray-700 mb-3">종목 검색(약 3분 가량 소요될 수 있습니다.)</label>
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
              onClick={fetchStockAndExplanationData}
              className="px-8 py-3 bg-indigo-600 text-white rounded-md shadow-md hover:bg-indigo-700 transition duration-300 ease-in-out font-semibold text-lg"
              disabled={loading || !ticker}
            >
              {loading ? '분석 중...' : '분석하기'}
            </button>
          </div>

          {error && (
            <p className="mt-4 text-red-600 bg-red-100 p-3 rounded-md border border-red-200">⚠️ {error}</p>
          )}

          {currentPrice !== null && predictedValue !== null && (
            <Card className="mb-6">
                <CardContent className="p-6">
                    <h2 className="text-xl font-bold mb-2">📈 예측 결과</h2>
                    <h2 className="text-xl font-bold mb-2">종목: {ticker.toUpperCase()}</h2>
                    <p className="text-xl text-gray-800 font-medium">최신 종가: <strong className="text-indigo-700">{currentPrice?.toFixed(2)}</strong></p>
                    <p className="text-xl text-gray-800 mt-3 font-medium">예측된 종가: <strong className="text-orange-600">{predictedValue?.toFixed(2)}</strong></p>
                    {recommendation && (
                        <p className={`mt-4 text-xl font-bold text-white px-5 py-2 inline-block rounded-full shadow-md ${
                            recommendation === '추천' ? 'bg-green-500' :
                            recommendation === '비추천' ? 'bg-red-500' :
                            'bg-gray-500'
                        }`}>
                            예측에 따른 추천 결과: <span className="text-white">{recommendation}</span>
                        </p>
                    )}
                </CardContent>
            </Card>
          )}

          {chartData && (
            <Card className="mb-6">
              <CardContent className="p-6">
                <h2 className="text-xl font-bold mb-4">📊 주가 추이 및 예측</h2>
                <Line data={chartData} options={chartOptions} />
              </CardContent>
            </Card>
          )}

          {influenceFactors.length > 0 && (
            <Card className="mb-6">
              <CardContent className="p-6">
                <h2 className="text-xl font-bold mb-4">📌 영향 요인별 해석</h2>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>영향 요인</TableHead>
                      <TableHead>설명</TableHead>
                      <TableHead>방향</TableHead>
                      <TableHead>강도</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {influenceFactors.map((f, i) => (
                      <TableRow key={i}>
                        <TableCell>{f.factor}</TableCell>
                        <TableCell>{f.description}</TableCell>
                        <TableCell>{f.direction === '상승' ? '📈 상승' : '📉 하락'}</TableCell>
                        <TableCell>{"●".repeat(f.strength) + "○".repeat(3 - f.strength)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

        <Button onClick={() => window.location.href = '/'} className="w-fit mx-auto mt-8">홈으로 돌아가기</Button>
        </div>
      </section>
    </main>
  );
}