'use client';

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import Header from '@/components/Header';
import { useState } from "react";

const OpportunityPage = () => {
  const [search, setSearch] = useState("");

  const opportunities = [
    {
      title: "💡 2025 창업경진대회",
      description: "창의적인 아이디어를 가진 초기 창업팀을 위한 전국 규모 대회입니다. 상금과 멘토링 제공.",
      tags: ["대회", "아이디어", "멘토링"]
    },
    {
      title: "💰 청년 창업 지원금",
      description: "만 39세 이하 예비창업자를 위한 정부 초기 자금 지원. 사업계획서와 IR 발표를 준비하세요.",
      tags: ["정부지원", "청년", "자금"]
    },
    {
      title: "🎤 IR 데모데이 참가자 모집",
      description: "VC 앞에서 발표할 수 있는 실전 피칭 무대! 투자 유치 기회를 잡아보세요.",
      tags: ["IR", "VC", "발표"]
    },
    {
      title: "🌍 해외 진출 스타트업 지원사업",
      description: "K-Startup의 글로벌 진출을 위한 마케팅, 전시회, 항공료 지원. 해외 시장 진입을 고려 중이라면 필수!",
      tags: ["글로벌", "해외", "마케팅"]
    },
  ];

  const filtered = opportunities.filter(
    (item) =>
      item.title.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase()) ||
      item.tags.some(tag => tag.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="pt-20 px-6 max-w-6xl mx-auto space-y-8">
      <Header />

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <Input
          placeholder="🔍 기회, 태그 또는 키워드를 검색하세요"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-md border-2 border-blue-500 focus:ring-blue-500 shadow"
        />
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filtered.map((item, index) => (
          <Card
            key={index}
            className="border-blue-100 bg-blue-50 hover:bg-white hover:shadow-lg transition cursor-pointer"
            onClick={() => alert(`${item.title}\n\n${item.description}`)}
          >
            <CardContent className="p-5">
              <h3 className="text-lg font-bold mb-2 text-blue-900">{item.title}</h3>
              <p className="text-sm text-gray-700 mb-2">{item.description}</p>
              <div className="flex flex-wrap gap-2">
                {item.tags.map((tag, idx) => (
                  <span
                    key={idx}
                    className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default OpportunityPage;
