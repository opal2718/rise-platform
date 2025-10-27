'use client';

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import Header from '@/components/Header';
import { useEffect, useState } from "react";
// import { supabase } from '../../../lib/supabaseClient'; // Supabase 관련 코드 주석 처리 또는 삭제

type Post = {
  id: number; // Cloud SQL에서 id가 자동으로 생성됩니다.
  title: string;
  text: string;
  userID: string;
  created_at: string; // 타임스탬프 필드 추가
};

const StartupCommunityPage = () => {
  const [search, setSearch] = useState("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [posts, setPosts] = useState<Post[]>([]);

  // 백엔드 API URL
  const API_BASE_URL = "http://34.16.110.5:7000/api"; // 백엔드 서버 주소

  // Fetch posts from backend API
  const fetchPosts = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/posts`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setPosts(data);
    } catch (error) {
      console.error('게시글 불러오기 실패:', error);
    }
  };

  // Add post to backend API and then refresh list
  const addPost = async (_user: string, _title: string, _text: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/posts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title: _title, userID: _user, text: _text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      // const newPost = await response.json(); // 새로 추가된 게시글 데이터를 받을 수도 있습니다.
      setTitle("");
      setContent("");
      await fetchPosts(); // 게시글 추가 후 목록 새로고침
    } catch (error) {
      console.error('게시글 추가 실패:', error);
    }
  };

  // Run once on component mount
  useEffect(() => {
    fetchPosts();
  }, []);

  const handleSubmit = () => {
    // 실제 사용자 ID를 사용하는 것이 좋습니다. 여기서는 임시로 "TESTID" 사용
    addPost("TESTID", title, content);
  };

  const filteredPosts = posts.filter(
    (post) =>
      post.title.toLowerCase().includes(search.toLowerCase()) ||
      post.text.toLowerCase().includes(search.toLowerCase()) ||
      post.userID.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="pt-22 px-6 max-w-4xl mx-auto grid gap-6">
      <Header />

      <Input
        placeholder="검색어를 입력하세요"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="max-w-sm"
      />

      <Card>
        <CardContent className="p-6 grid gap-4">
          <h2 className="text-xl font-bold">✏️ 새 글 작성</h2>
          <Input placeholder="제목을 입력하세요" value={title} onChange={(e) => setTitle(e.target.value)} />
          <Textarea placeholder="내용을 입력하세요..." rows={4} value={content} onChange={(e) => setContent(e.target.value)} />
          <Button className="w-fit" onClick={handleSubmit}>작성 완료</Button>
        </CardContent>
      </Card>

      <div className="grid gap-4">
        {filteredPosts.map((post) => ( // key prop에 id 사용
          <Card key={post.id}>
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold">{post.title}</h3>
              {/* <p className="text-sm text-muted-foreground mb-1">by {post.userID}</p> */}
              <p>{post.text}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default StartupCommunityPage;