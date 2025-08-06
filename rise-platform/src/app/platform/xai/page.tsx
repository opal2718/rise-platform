'use client';

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


const PredictExplainPage = () => {
  const pastData = [103, 105, 106, 104, 107, 108, 109];
  const predicted = [110, 112, 114, 115, 116];

  const labels = [...Array(pastData.length + predicted.length).keys()].map(
    (i) => `t+${i - pastData.length + 1}`
  );

  const data = {
    labels,
    datasets: [
      {
        label: "과거 주가",
        data: [...pastData, null, null, null, null, null],
        borderColor: "gray",
        tension: 0.3,
      },
      {
        label: "예측 주가",
        data: [...Array(pastData.length).fill(null), ...predicted],
        borderDash: [5, 5],
        borderColor: "blue",
        tension: 0.3,
      },
    ],
  };

  const recommendation = predicted[4] > pastData[pastData.length - 1] + 2
    ? "상승 추천"
    : predicted[4] < pastData[pastData.length - 1] - 2
    ? "하락 비추천"
    : "중립";

  const influenceFactors = [
    {
      factor: "최근 3일 주가 추세",
      description: "주가가 지속적으로 상승하고 있음",
      direction: "상승",
      strength: 2,
    },
    {
      factor: "거래량 변화율",
      description: "평균 대비 거래량이 20% 증가",
      direction: "상승",
      strength: 3,
    },
    {
      factor: "RSI 과열 지표",
      description: "RSI = 78 → 과매수 구간",
      direction: "하락",
      strength: 3,
    },
  ];

  return (
    <div className="p-6 pt-22 grid gap-6 max-w-3xl mx-auto">
      <Header />
      <Card>
        <CardContent className="p-6">
          <h2 className="text-xl font-bold mb-2">📈 예측 결과</h2>
          <h2 className="text-xl font-bold mb-2">종목: TSLA(테슬라)</h2>
          <p>예측된 종가: <b>{+predicted[4]}</b></p>
          <p>추천: <b className="text-lg">{recommendation}</b></p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-6">
          <h2 className="text-xl font-bold mb-4">📊 예측 기반 그래프</h2>
          <Line data={data} />
        </CardContent>
      </Card>

      <Card>
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

      <Button className="w-fit mx-auto">홈으로 돌아가기</Button>
    </div>
  );
};

export default PredictExplainPage;
