const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/api/analyse', async (req, res) => {
  try {
    const { text } = req.body;

    // AI 서버에 요청
    const aiRes = await axios.post('http://localhost:8000/predict', { text });

    // AI 응답 전달
    res.json({ result: aiRes.data });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'AI 서버 호출 실패' });
  }
});

app.post('/api/predict', async (req, res) => {
  const { text: A } = req.body;

  try {
    console.log("hihihi");
    // Step 1: A → AI 서버 1
    const { data: B } = await axios.post('http://0.0.0.0:8000/process', { input: "hiihiihihi" });

    // Step 2: B → AI 서버 2
    const { data: C } = await axios.post('http://localhost:8001/process', { input: B.output });

    // Step 3: C → D (메인 백엔드에서 가공)
    const D = (C); // 이 함수는 직접 정의해야 해

    // 반환
    res.json({ result: D });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "예측 처리 중 오류 발생" });
  }
});


app.listen(3001, '0.0.0.0', () => {
  console.log('메인 백엔드 서버 실행 중');
});
