const express = require('express');
const { Pool } = require('pg'); // 또는 MySQL의 경우 const mysql = require('mysql2');
const cors = require('cors');

const app = express();
const port = 7000; // 백엔드 서버가 실행될 포트

app.use(cors());
app.use(express.json()); // JSON 요청 본문 파싱

// Google Cloud SQL 연결 설정
const pool = new Pool({
  user: 'postgres', // Cloud SQL 사용자 이름
  host: '34.9.33.206', // Cloud SQL 인스턴스의 공용 IP (임시)
                                     // 프로덕션에서는 Cloud SQL Proxy 또는 비공개 IP 사용 권장
  database: 'postgres', // Cloud SQL 데이터베이스 이름
  password: 'wegoUP1234!', // Cloud SQL 사용자 비밀번호
  port: 5432, // PostgreSQL 기본 포트 (MySQL은 3306)
  ssl: { // <--- 이 부분을 추가합니다!
    rejectUnauthorized: false // 개발용: SSL 인증서 검증 무시 (프로덕션에서는 보안 강화 필요)
  }
});

// 연결 테스트
pool.on('connect', () => {
  console.log('Google Cloud SQL connected!');
});

pool.on('error', (err) => {
  console.error('Unexpected error on idle client', err);
  process.exit(-1);
});

// 게시글 테이블 생성 (한 번만 실행)
async function createPostsTable() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS community_posts (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        text TEXT NOT NULL,
        userID VARCHAR(255) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      );
    `);
    console.log('community_posts table ensured.');
  } catch (err) {
    console.error('Error creating table:', err);
  }
}
createPostsTable();


// 모든 게시글 가져오기 API
app.get('/api/posts', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM community_posts ORDER BY created_at DESC');
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching posts:', err);
    res.status(500).json({ error: 'Failed to fetch posts' });
  }
});

// 새 게시글 추가 API
app.post('/api/posts', async (req, res) => {
  const { title, userID, text } = req.body;
  try {
    const result = await pool.query(
      'INSERT INTO community_posts (title, userID, text) VALUES ($1, $2, $3) RETURNING *',
      [title, userID, text]
    );
    res.status(201).json(result.rows[0]);
  } catch (err) {
    console.error('Error adding post:', err);
    res.status(500).json({ error: 'Failed to add post' });
  }
});

app.listen(port, () => {
  console.log(`Backend server listening at http://34.16.110.5:${port}`);
});