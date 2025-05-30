'use client';

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import Header from '@/components/Header';
import { useState } from "react";

const StartupCommunityPage = () => {
  const [search, setSearch] = useState("");

  const posts = [
    {
      author: "이하람",
      title: "창업 아이디어 피드백 부탁드려요!",
      content: "AI를 활용한 개인 맞춤 건강관리 서비스인데, 시장성에 대한 의견이 궁금합니다.",
    },
    {
      author: "김은지",
      title: "IR 피치덱 공유합니다",
      content: "투자 발표를 앞두고 있는데 혹시 보완할 부분 있을까요?",
    },
  ];

  const filteredPosts = posts.filter(
    (post) =>
      post.title.toLowerCase().includes(search.toLowerCase()) ||
      post.content.toLowerCase().includes(search.toLowerCase()) ||
      post.author.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="pt-22 px-6 max-w-4xl mx-auto grid gap-6">
      <Header/>

      <Input
        placeholder="검색어를 입력하세요"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="max-w-sm"
      />

      <Card>
        <CardContent className="p-6 grid gap-4">
          <h2 className="text-xl font-bold">✏️ 새 글 작성</h2>
          <Input placeholder="제목을 입력하세요" />
          <Textarea placeholder="내용을 입력하세요..." rows={4} />
          <Button className="w-fit">작성 완료</Button>
        </CardContent>
      </Card>

      <div className="grid gap-4">
        {filteredPosts.map((post, index) => (
          <Card key={index}>
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold">{post.title}</h3>
              <p className="text-sm text-muted-foreground mb-1">by {post.author}</p>
              <p>{post.content}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default StartupCommunityPage;
