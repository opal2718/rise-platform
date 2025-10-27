'use client'

import Header from '@/components/Header';

import { useEffect, useState } from "react";

export default function Home() {
  const [, setData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      console.log("hii"); // 여기서는 찍혀야 함
      const res = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'A' }),
      });
      const json = await res.json();
      setData(json.result);
      console.log(json.result);
    }

    fetchData();
  }, []);
  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 bg-[#f5f7f9]">
      <Header />

      {/* Hero Section */}
      <section className="text-center py-24">
        <h1 className="text-5xl font-semibold text-gray-700 tracking-tight mb-4">AI와 함께 미래를 예측하다 – RISE</h1>
        <p className="text-lg text-gray-500">주가 예측, 창업 지원, 경제 교육까지 한 곳에서</p>
      </section>

      {/* About RISE */}
      <section className="py-16 max-w-5xl text-center">
        <h2 className="text-3xl font-semibold text-gray-700 mb-8">RISE의 핵심 기능</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-10">
          <div>
            <div className="text-4xl mb-3">📈</div>
            <h3 className="text-lg font-medium text-gray-700">AI 기반 주가 예측</h3>
            <p className="text-sm text-gray-500 mt-1">River의 ARFR을 활용한 유연한 예측</p>
          </div>
          <div>
            <div className="text-4xl mb-3">🧠</div>
            <h3 className="text-lg font-medium text-gray-700">설명 가능한 AI</h3>
            <p className="text-sm text-gray-500 mt-1">SHAP을 통해 예측 근거를 시각화</p>
          </div>
          <div>
            <div className="text-4xl mb-3">💡</div>
            <h3 className="text-lg font-medium text-gray-700">창업 커뮤니티 지원</h3>
            <p className="text-sm text-gray-500 mt-1">창업 정보, 커뮤니티, 자료 제공</p>
          </div>
        </div>
      </section>
      <footer className="w-full bg-[#e6eaee] text-gray-600 text-sm py-6 mt-12">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-2">
          <p>&copy; 2025 RISE Platform. All rights reserved.</p>
          <div className="flex gap-4">
            <a href="mailto:rise@riseplatform.kr" className="hover:underline">Contact</a>
            <a href="/terms" className="hover:underline">Terms</a>
            <a href="/privacy" className="hover:underline">Privacy</a>
          </div>
        </div>
      </footer>

    </main>
    
  );
}
