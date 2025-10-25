const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/api/analyse', async (req, res) => {
  try {
    const { text } = req.body;

    // Input validation for the text field
    if (!text) {
      console.error("Missing 'text' field in /api/analyse request body.");
      return res.status(400).json({ error: "Missing 'text' field in request body" });
    }

    // AI 서버에 요청
    const aiRes = await axios.post('http://ai1:8000/predict', { text });

    // AI 응답 전달
    res.json({ result: aiRes.data });

  } catch (err) {
    // Check if the error is an AxiosError (meaning a response was received, but it was an error status)
    if (axios.isAxiosError(err) && err.response) {
      const statusCode = err.response.status;
      const errorData = err.response.data; // This is where ai1's HTTPException detail will be

      console.error(`Error calling AI server (Status: ${statusCode}):`, errorData);
      
      // If ai1 sends back a specific error message in its response body, forward it
      // The `detail` key is what FastAPI's HTTPException uses by default.
      if (errorData && typeof errorData === 'object' && errorData.detail) {
        return res.status(statusCode).json({ 
          error: `AI server failed: ${errorData.detail}`,
          aiServerError: errorData.detail // Provide the raw detail as well
        });
      } else if (typeof errorData === 'string') {
        // Sometimes FastAPI might return a plain string for 500s or other errors
        return res.status(statusCode).json({
          error: `AI server failed: ${errorData}`,
          aiServerError: errorData
        });
      } else {
        // Fallback for unexpected error data format
        return res.status(statusCode).json({ 
          error: `AI server returned an error (Status: ${statusCode}). Details: ${JSON.stringify(errorData)}`,
          aiServerError: errorData
        });
      }
    } else if (axios.isAxiosError(err) && err.request) {
      // The request was made but no response was received (e.g., network error, ai1 container not running)
      console.error("No response received from AI server:", err.message);
      return res.status(503).json({ 
        error: 'AI server is unreachable or did not respond.',
        detailedError: err.message
      });
    } else {
      // Something else happened in setting up the request or a non-Axios error occurred
      console.error("Unexpected error during AI server call:", err);
      return res.status(500).json({ 
        error: 'An unexpected error occurred while calling the AI server.',
        detailedError: err.message 
      });
    }
  }
});

app.post('/api/predict', async (req, res) => {
  const { text: A } = req.body;

  try {
    console.log("hihihi");
    // Step 1: A → AI 서버 1
    const { data: B } = await axios.post('http://ai1:8000/process', { input: "hiihiihihi" });

    // Step 2: B → AI 서버 2
    const { data: C } = await axios.post('http://ai2:8001/process', { input: B.output });

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
