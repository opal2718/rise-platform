'use client';

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import Header from '@/components/Header';
import { useEffect, useState } from "react";
import { supabase } from '../../../lib/supabaseClient';

type Post = {
  title: string;
  text: string;
  userID: string;
};

const StartupCommunityPage = () => {
  const [search, setSearch] = useState("");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [posts, setPosts] = useState<Post[]>([]);

  // Fetch posts from Supabase
  const fetchPosts = async () => {
    const { data, error } = await supabase.from('community posts').select('*');
    if (error) {
      console.error('불러오기 실패:', error.message);
    } else {
      setPosts(data.reverse()); // 최신 글이 위로
    }
  };

  // Add post and then refresh list
  const addPost = async (_user: string, _title: string, _text: string) => {
    const { error } = await supabase
      .from('community posts')
      .insert([{ title: _title, userID: _user, text: _text }]);

    if (error) {
      console.error('삽입 실패:', error.message);
    } else {
      setTitle("");
      setContent("");
      await fetchPosts(); // Re-fetch posts instead of reloading
    }
  };

  // Run once on component mount
  useEffect(() => {
    fetchPosts();
  }, []);

  const handleSubmit = () => {
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
        {filteredPosts.map((post, index) => (
          <Card key={index}>
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold">{post.title}</h3>
              <p className="text-sm text-muted-foreground mb-1">by {post.userID}</p>
              <p>{post.text}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default StartupCommunityPage;
