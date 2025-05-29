export default function Home() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-50">
      {/* Hero Section */}
      <section className="text-center py-16">
        <h1 className="text-5xl font-bold text-gray-800 mb-4">AI와 함께 미래를 예측하다 – RISE</h1>
        <p className="text-xl text-gray-600 mb-8">주가 예측, 창업 지원, 경제 교육까지 한 곳에서</p>
        <div className="flex flex-wrap justify-center gap-4">
          <button className="px-6 py-3 bg-blue-600 text-white rounded-2xl shadow hover:bg-blue-700">주가 예측 시작하기</button>
          <button className="px-6 py-3 bg-green-500 text-white rounded-2xl shadow hover:bg-green-600">회원가입</button>
          <button className="px-6 py-3 bg-purple-500 text-white rounded-2xl shadow hover:bg-purple-600">창업 커뮤니티</button>
        </div>
      </section>

      {/* About RISE */}
      <section className="py-12 max-w-5xl text-center">
        <h2 className="text-3xl font-semibold mb-8">RISE의 핵심 기능</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          <div>
            <div className="text-4xl mb-2">📈</div>
            <h3 className="text-xl font-semibold">AI 기반 주가 예측</h3>
            <p className="text-gray-600">LSTM, XGBoost 등을 활용한 정확한 예측</p>
          </div>
          <div>
            <div className="text-4xl mb-2">🧠</div>
            <h3 className="text-xl font-semibold">설명 가능한 AI</h3>
            <p className="text-gray-600">SHAP, KAN을 통해 예측 근거를 시각화</p>
          </div>
          <div>
            <div className="text-4xl mb-2">💡</div>
            <h3 className="text-xl font-semibold">창업 커뮤니티 지원</h3>
            <p className="text-gray-600">청소년 창업 정보, 커뮤니티, 자료 제공</p>
          </div>
        </div>
      </section>

      {/* Placeholder for more sections */}
      <section className="py-20 text-center text-gray-400">
        <p>More coming soon...</p>
      </section>
    </main>
  );
}
