"""
LLM 채팅 엔드포인트
OpenAI GPT 모델을 사용하여 책 내용 기반 AI 채팅 기능을 제공합니다.
"""
import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Java Backend RAG 엔드포인트 (내부 호출용)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")


class ChatType(str, Enum):
    DEFINITION = "DEFINITION"
    CONTENT_QA = "CONTENT_QA"
    SUMMARY = "SUMMARY"
    QUIZ = "QUIZ"
    CHIT_CHAT = "CHIT_CHAT"


# ==================== Request/Response DTOs ====================

class ClassifyQuestionRequest(BaseModel):
    user_message: str = Field(..., description="사용자 질문")


class ClassifyQuestionResponse(BaseModel):
    chat_type: ChatType = Field(..., description="분류된 질문 유형")
    confidence: float = Field(..., description="분류 신뢰도 (0~1)")


class GenerateAnswerRequest(BaseModel):
    chapter_id: int = Field(..., description="챕터 ID")
    user_message: str = Field(..., description="사용자 질문")
    chat_type: Optional[ChatType] = Field(None, description="채팅 유형 (없으면 자동 분류)")
    current_paragraph_id: Optional[str] = Field(None, description="현재 읽고 있는 문단 ID")
    chat_history: Optional[List[dict]] = Field(default=[], description="이전 대화 기록")


class SourceReference(BaseModel):
    paragraph_ids: List[str] = Field(..., description="출처 문단 ID 목록")
    content_preview: str = Field(..., description="출처 내용 미리보기")
    similarity: float = Field(..., description="유사도 점수")


class GenerateAnswerResponse(BaseModel):
    answer: str = Field(..., description="AI 답변")
    chat_type: ChatType = Field(..., description="사용된 채팅 유형")
    sources: Optional[List[SourceReference]] = Field(None, description="RAG 출처 (있는 경우)")
    token_count: Optional[int] = Field(None, description="사용된 토큰 수")


# ==================== Helper Functions ====================

async def call_openai(messages: List[dict], model: str = "gpt-4o", max_tokens: int = 1024) -> dict:
    """OpenAI API 호출 헬퍼 함수"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY가 설정되지 않았습니다.")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI API 오류: {response.text}")
        
        return response.json()


async def classify_question_internal(user_message: str) -> tuple[ChatType, float]:
    """질문 유형 자동 분류 (GPT-4o-mini 사용)"""
    
    system_prompt = """당신은 독서 AI 어시스턴트의 질문 분류기입니다.
사용자의 질문을 다음 5가지 유형 중 하나로 분류하세요:

1. DEFINITION: 단어/용어의 뜻을 묻는 질문
   예: "이 단어 뜻이 뭐야?", "푸른의 의미는?"

2. CONTENT_QA: 책 내용에 대한 질문이나 심층 분석
   예: "주인공이 왜 화를 낸 거야?", "이 복선은 어떻게 회수돼?"

3. SUMMARY: 요약 요청
   예: "지금까지 내용 요약해줘", "이 챕터 3줄 요약"

4. QUIZ: 퀴즈/문제 출제 요청
   예: "내가 잘 이해했는지 퀴즈 내줘"

5. CHIT_CHAT: 일상 대화나 책과 무관한 질문
   예: "안녕", "고마워", "너 똑똑하네"

반드시 다음 JSON 형식으로만 응답하세요:
{"type": "CONTENT_QA", "confidence": 0.95}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        result = await call_openai(messages, model="gpt-4o-mini", max_tokens=50)
        content = result["choices"][0]["message"]["content"]
        
        # JSON 파싱
        import json
        parsed = json.loads(content)
        chat_type = ChatType(parsed["type"])
        confidence = float(parsed["confidence"])
        
        return chat_type, confidence
    except Exception as e:
        print(f"⚠️ 질문 분류 실패, 기본값(CONTENT_QA) 사용: {e}")
        return ChatType.CONTENT_QA, 0.5


async def fetch_rag_context(chapter_id: int, query: str) -> List[dict]:
    """Backend의 RAG 검색 API 호출"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 내부 RAG 검색 호출 (ChapterVectorRagController)
            response = await client.get(
                f"{BACKEND_BASE_URL}/v1/chapters/{chapter_id}/rag-search",
                params={"query": query}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ RAG 검색 실패: {response.status_code}")
                return []
    except Exception as e:
        print(f"⚠️ RAG 검색 중 오류: {e}")
        return []


def build_chat_prompt(chat_type: ChatType, user_message: str, rag_context: List[dict] = None) -> str:
    """채팅 유형에 따른 시스템 프롬프트 생성"""
    
    base_prompt = """당신은 친절하고 지식이 풍부한 독서 AI 어시스턴트입니다.
사용자가 읽고 있는 책의 내용을 바탕으로 질문에 답변해 주세요.
한국어로 자연스럽고 친근하게 대화하세요."""

    if chat_type == ChatType.DEFINITION:
        return f"""{base_prompt}

사용자가 단어나 용어의 뜻을 물었습니다.
- 해당 단어의 사전적 의미를 설명하세요.
- 책의 문맥에서 어떤 의미로 사용되었는지도 설명해 주세요."""

    elif chat_type == ChatType.CONTENT_QA:
        context_text = ""
        if rag_context:
            context_text = "\n\n[참고할 책 내용]\n"
            for i, ctx in enumerate(rag_context, 1):
                content = ctx.get("contentText", "")[:500]
                context_text += f"\n{i}. {content}\n"
        
        return f"""{base_prompt}

사용자가 책 내용에 대해 질문했습니다.
- 제공된 책 내용을 참고하여 정확하게 답변하세요.
- 책에 나오지 않는 내용은 추측임을 명시하세요.
{context_text}"""

    elif chat_type == ChatType.SUMMARY:
        context_text = ""
        if rag_context:
            context_text = "\n\n[요약할 내용]\n"
            for ctx in rag_context:
                context_text += ctx.get("contentText", "") + "\n"
        
        return f"""{base_prompt}

사용자가 요약을 요청했습니다.
- 핵심 내용을 간결하게 정리하세요.
- 중요한 사건이나 인물 관계를 포함하세요.
{context_text}"""

    elif chat_type == ChatType.QUIZ:
        return f"""{base_prompt}

사용자가 퀴즈를 요청했습니다.
- 책 내용을 바탕으로 이해도를 테스트할 수 있는 질문을 만드세요.
- 객관식 또는 단답형 문제를 2~3개 출제하세요.
- 정답은 바로 알려주지 말고, 사용자가 답변한 후에 피드백하세요."""

    elif chat_type == ChatType.CHIT_CHAT:
        return f"""{base_prompt}

사용자가 일상적인 대화를 하고 있습니다.
- 친근하고 자연스럽게 대화하세요.
- 적절한 경우 책 읽기와 관련된 이야기로 자연스럽게 연결하세요."""

    return base_prompt


# ==================== API Endpoints ====================

@router.post("/classify-question", response_model=ClassifyQuestionResponse)
async def classify_question(request: ClassifyQuestionRequest):
    """
    사용자 질문을 5가지 유형 중 하나로 분류합니다.
    GPT-4o-mini를 사용하여 빠르고 저렴하게 분류합니다.
    """
    chat_type, confidence = await classify_question_internal(request.user_message)
    return ClassifyQuestionResponse(chat_type=chat_type, confidence=confidence)


@router.post("/generate-answer", response_model=GenerateAnswerResponse)
async def generate_answer(request: GenerateAnswerRequest):
    """
    사용자 질문에 대한 AI 답변을 생성합니다.
    
    1. chat_type이 없으면 자동 분류
    2. RAG가 필요한 유형이면 벡터 검색 수행
    3. GPT-4o로 답변 생성
    4. 출처 정보와 함께 반환
    """
    # 1. 질문 유형 결정
    if request.chat_type:
        chat_type = request.chat_type
    else:
        chat_type, _ = await classify_question_internal(request.user_message)
    
    # 2. RAG 필요 여부 판단 및 컨텍스트 수집
    rag_context = []
    sources = []
    
    if chat_type in [ChatType.CONTENT_QA, ChatType.SUMMARY]:
        rag_context = await fetch_rag_context(request.chapter_id, request.user_message)
        
        # 출처 정보 변환
        for ctx in rag_context:
            sources.append(SourceReference(
                paragraph_ids=ctx.get("paragraphIds", []),
                content_preview=ctx.get("contentText", "")[:200] + "...",
                similarity=ctx.get("similarity", 0.0)
            ))
    
    # 3. 프롬프트 구성
    system_prompt = build_chat_prompt(chat_type, request.user_message, rag_context)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 대화 기록 추가
    if request.chat_history:
        for msg in request.chat_history[-10:]:  # 최근 10개만
            messages.append(msg)
    
    messages.append({"role": "user", "content": request.user_message})
    
    # 4. GPT 호출
    result = await call_openai(messages, model="gpt-4o", max_tokens=1024)
    
    answer = result["choices"][0]["message"]["content"]
    token_count = result.get("usage", {}).get("total_tokens")
    
    return GenerateAnswerResponse(
        answer=answer,
        chat_type=chat_type,
        sources=sources if sources else None,
        token_count=token_count
    )


# SSE 스트리밍 엔드포인트
from fastapi.responses import StreamingResponse
import asyncio

async def call_openai_stream(messages: List[dict], model: str = "gpt-4o", max_tokens: int = 1024):
    """OpenAI API 스트리밍 호출"""
    if not OPENAI_API_KEY:
        yield "data: OPENAI_API_KEY가 설정되지 않았습니다.\n\n"
        return
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", OPENAI_API_URL, headers=headers, json=payload) as response:
            if response.status_code != 200:
                yield f"data: OpenAI API 오류: {response.status_code}\n\n"
                return
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        pass


@router.post("/generate-answer-stream")
async def generate_answer_stream(request: GenerateAnswerRequest):
    """
    사용자 질문에 대한 AI 답변을 SSE 스트리밍으로 생성합니다.
    chatType은 자동 분류됩니다.
    """
    async def event_generator():
        # 1. 질문 유형 결정
        if request.chat_type:
            chat_type = request.chat_type
        else:
            chat_type, _ = await classify_question_internal(request.user_message)
        
        # 2. RAG 필요 여부 판단 및 컨텍스트 수집
        rag_context = []
        
        if chat_type in [ChatType.CONTENT_QA, ChatType.SUMMARY]:
            rag_context = await fetch_rag_context(request.chapter_id, request.user_message)
        
        # 3. 프롬프트 구성
        system_prompt = build_chat_prompt(chat_type, request.user_message, rag_context)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 대화 기록 추가
        if request.chat_history:
            for msg in request.chat_history[-10:]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": request.user_message})
        
        # 4. GPT 스트리밍 호출
        async for chunk in call_openai_stream(messages, model="gpt-4o", max_tokens=1024):
            yield chunk
            await asyncio.sleep(0)  # 비동기 yield 허용
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

