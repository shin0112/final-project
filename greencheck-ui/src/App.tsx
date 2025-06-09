import { useState } from "react";
import styled from "styled-components";

type ResultData = {
  raw: string;
  data: {
    judgement: string;
    reason: string;
    law: string;
    solution: string;
  };
};

function App() {
  const [article, setArticle] = useState("");
  const [result, setResult] = useState<ResultData | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ article, ct: "없음" }),
      });
      const data = await res.json();
      console.log("[서버 응답]", data);
      setResult(data);
    } catch (err) {
      alert("서버 호출 실패: " + err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container>
      <Card>
        <Title>🟢 GreenCheck: 그린워싱 탐지기</Title>

        <TextArea
          placeholder="기사 내용을 입력하세요"
          value={article}
          onChange={(e) => setArticle(e.target.value)}
        />

        <Button onClick={handleRun} disabled={loading}>
          {loading ? "분석 중..." : "분석 실행"}
        </Button>

        {result && (
          <Section>
            <SectionTitle>✅ 판단</SectionTitle>
            <SectionContent>{result.data.judgement || "(없음)"}</SectionContent>

            <SectionTitle>📌 근거</SectionTitle>
            <SectionContent>{result.data.reason || "(없음)"}</SectionContent>

            <SectionTitle>📚 법률</SectionTitle>
            <SectionContent>{result.data.law || "(없음)"}</SectionContent>

            <SectionTitle>🛠 해결방안</SectionTitle>
            <SectionContent>{result.data.solution || "(없음)"}</SectionContent>
          </Section>
        )}
      </Card>
    </Container>
  );
}

export default App;

const Container = styled.div`
  min-height: 100vh;
  background-color: #f3f4f6;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 5rem 2rem;

  // ✅ 가운데 고정 폭 추가
  max-width: 100%;
  box-sizing: border-box;
`;

const Card = styled.div`
  background-color: white;
  padding: 3rem;
  max-width: 1100px;
  width: 100%;
  border-radius: 1.5rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
  justify-content: center;
  align-items: center;
  display: flex;
  flex-direction: column;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 800;
  color: #15803d;
  text-align: center;
  margin-bottom: 2.5rem;
`;

const TextArea = styled.textarea`
  width: 95%;
  min-height: 300px;
  padding: 1.5rem;
  font-size: 1.2rem;
  border: 1px solid #d1d5db;
  border-radius: 0.75rem;
  margin-bottom: 2rem;
  resize: vertical;
  justify-content: center;
  align-items: center;
`;

const Button = styled.button`
  background-color: #15803d;
  color: white;
  font-weight: 600;
  padding: 1rem 2.5rem;
  font-size: 1.125rem;
  border: none;
  border-radius: 0.75rem;
  cursor: pointer;
  display: block;
  margin: 0 auto;

  &:hover {
    background-color: #166534;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const Section = styled.div`
  margin-top: 3rem;
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: #111827;
`;

const SectionContent = styled.p`
  font-size: 1.125rem;
  line-height: 1.75;
  color: #374151;
  white-space: pre-wrap;
`;
