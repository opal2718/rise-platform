'use client';

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";
import Header from '@/components/Header';
import { useState } from "react";

const BeginnerGuidePage = () => {
  const [search, setSearch] = useState("");
  const [searchField, setSearchField] = useState("all");

  const articles = [
    {
      title: "사업자 등록 절차",
      summary: "창업 초기에 꼭 필요한 사업자 등록 방법과 필요한 서류 안내.",
    },
    {
      title: "지식재산권 기초 가이드",
      summary: "특허, 상표, 디자인 등록의 차이점과 등록 방법에 대해 설명합니다.",
    },
    {
      title: "스타트업을 위한 계약서 체크리스트",
      summary: "동업계약, 투자계약 등 꼭 확인해야 할 조항들을 정리했습니다.",
    },
    {
      title: "초기 자금 조달 방법",
      summary: "정부지원, 엔젤투자, 크라우드펀딩 등 다양한 초기 자금 유치 전략.",
    },
  ];

  const filteredArticles = articles.filter((article) => {
    const term = search.toLowerCase();
    if (searchField === "title") return article.title.toLowerCase().includes(term);
    if (searchField === "summary") return article.summary.toLowerCase().includes(term);
    return (
      article.title.toLowerCase().includes(term) ||
      article.summary.toLowerCase().includes(term)
    );
  });

  return (
    <div className="pt-20 px-6 max-w-6xl mx-auto grid gap-6">
      <Header />

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <Input
          placeholder="🔍 법률 정보 또는 키워드를 검색하세요"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm border-2 border-primary shadow"
        />
        <Select onValueChange={(val) => setSearchField(val)} defaultValue="all">
          <SelectTrigger className="w-[160px]">
            {searchField === "all" && "전체 검색"}
            {searchField === "title" && "제목만"}
            {searchField === "summary" && "내용만"}
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">전체 검색</SelectItem>
            <SelectItem value="title">제목만</SelectItem>
            <SelectItem value="summary">내용만</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {filteredArticles.map((article, index) => (
          <Button
            key={index}
            variant="outline"
            className="text-left h-full w-full p-0 hover:shadow-md"
            onClick={() => alert(`${article.title}\n\n${article.summary}`)}
          >
            <Card className="w-full h-full">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold">{article.title}</h3>
                <p className="text-muted-foreground text-sm mt-1">{article.summary}</p>
              </CardContent>
            </Card>
          </Button>
        ))}
      </div>
    </div>
  );
};

export default BeginnerGuidePage;
