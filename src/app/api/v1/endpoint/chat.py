from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
import openai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

router = APIRouter()

# OpenAI 클라이언트 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # 로컬 개발 환경 등에서 키가 없을 수 있으므로 경고만 로그에 남기거나, 실제 호출 시 에러 처리
    pass

client = openai.OpenAI(api_key=api_key)

class ChatContext(BaseModel):
    paragraph_id: Optional[str] = None
    content: Optional[str] = None

class ClassifyRequest(BaseModel):
    user_msg: str
    chapter_id: Optional[int] = None
    current_paragraph_content: Optional[str] = None

class ClassifyResponse(BaseModel):
    chat_type: str  # DEFINITION, CONTENT_QA, SUMMARY, QUIZ, CHIT_CHAT

class GenerateRequest(BaseModel):
    user_msg: str
    chat_type: str
    rag_context: Optional[str] = None
    previous_messages: Optional[List[dict]] = None # [{"role": "user", "content": "..."}, ...]

class GenerateResponse(BaseModel):
    response: str
    token_usage: Optional[int] = 0

@router.post("/chat/classify", response_model=ClassifyResponse)
async def classify_message(request: ClassifyRequest):
    """
    사용자 메시지의 의도를 분류합니다.
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured")

    system_prompt = """
    You are an AI assistant for a book reader application 'ReadSync'.
    Classify the user's message into one of the following categories:
    
    1. DEFINITION: Asking for the meaning of a word, term, or concept. (e.g., "What does 'ephemeral' mean?", "Explain this word")
    2. CONTENT_QA: Asking questions about the book's specific content, plot, characters, or hidden meanings. Requires knowledge of the book. (e.g., "Why did the protagonist cry?", "What happened in chapter 1?")
    3. SUMMARY: Asking to summarize the current chapter or section. (e.g., "Summarize this chapter", "What is this page about?")
    4. QUIZ: Asking to generate a quiz or test about the book. (e.g., "Give me a quiz", "Test my understanding")
    5. CHIT_CHAT: General conversation, greetings, or questions unrelated to the book content. (e.g., "Hello", "You are smart", "Thank you")
    
    Return ONLY the category name. Do not add any other text.
    """
    
    user_content = f"User Message: {request.user_msg}"
    if request.current_paragraph_content:
        user_content += f"\n(Context: User is reading this paragraph: '{request.current_paragraph_content[:100]}...')"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # 가벼운 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_completion_tokens=20 
        )
        
        chat_type = completion.choices[0].message.content.strip().upper()
        
        # 유효성 검사 (혹시 모를 환각 방지)
        valid_types = ["DEFINITION", "CONTENT_QA", "SUMMARY", "QUIZ", "CHIT_CHAT"]
        if chat_type not in valid_types:
            # 기본값으로 CONTENT_QA 또는 CHIT_CHAT 설정
            if "MEAN" in chat_type or "DEFINE" in chat_type:
                chat_type = "DEFINITION"
            else:
                chat_type = "CONTENT_QA"
                
        return ClassifyResponse(chat_type=chat_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    주어진 컨텍스트(RAG 결과 등)와 채팅 타입을 바탕으로 답변을 생성합니다.
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured")

    system_prompt = "You are a helpful AI reading assistant named 'ReadSync AI'. Answer in Korean."

    if request.chat_type == "DEFINITION":
        system_prompt += " Explain the term clearly and concisely based on standard dictionary definitions. If the term has a specific meaning in the book's context provided, prioritize that."
    elif request.chat_type == "CONTENT_QA":
        system_prompt += " Answer the question based ONLY on the provided Context. If the answer is not in the Context, say '제공된 내용에서는 답을 찾을 수 없습니다.' Cite the source paragraph if possible."
    elif request.chat_type == "SUMMARY":
        system_prompt += " Summarize the provided text concisely."
    elif request.chat_type == "QUIZ":
        system_prompt += " Generate a multiple-choice quiz based on the context. Provide the question, 4 options, and the answer."
    elif request.chat_type == "CHIT_CHAT":
        system_prompt += " Engage in a friendly, polite conversation. Do not make up facts about the book if you don't know."

    messages = [{"role": "system", "content": system_prompt}]
    
    # 이전 대화 내역 추가 (백엔드에서 이미 필터링해서 보내지만, 안전장치로 유지)
    if request.previous_messages:
        messages.extend(request.previous_messages[-10:])
        
    # 현재 컨텍스트 및 질문 추가
    user_content = request.user_msg
    if request.rag_context:
        user_content = f"Context:\n{request.rag_context}\n\nQuestion: {request.user_msg}"
    
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # 비용 효율적인 모델 사용
            messages=messages,
            temperature=0.7
        )
        
        response_text = completion.choices[0].message.content
        token_usage = completion.usage.total_tokens if completion.usage else 0
        
        return GenerateResponse(response=response_text, token_usage=token_usage)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse

@router.post("/chat/generate/stream")
async def generate_response_stream(request: GenerateRequest):
    """
    답변을 SSE 스트림으로 반환합니다.
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured")

    system_prompt = "You are a helpful AI reading assistant named 'ReadSync AI'. Answer in Korean."

    if request.chat_type == "DEFINITION":
        system_prompt += " Explain the term clearly and concisely based on standard dictionary definitions. If the term has a specific meaning in the book's context provided, prioritize that."
    elif request.chat_type == "CONTENT_QA":
        system_prompt += " Answer the question based ONLY on the provided Context. If the answer is not in the Context, say '제공된 내용에서는 답을 찾을 수 없습니다.' Cite the source paragraph if possible."
    elif request.chat_type == "SUMMARY":
        system_prompt += " Summarize the provided text concisely."
    elif request.chat_type == "QUIZ":
        system_prompt += " Generate a multiple-choice quiz based on the context. Provide the question, 4 options, and the answer."
    elif request.chat_type == "CHIT_CHAT":
        system_prompt += " Engage in a friendly, polite conversation. Do not make up facts about the book if you don't know."

    messages = [{"role": "system", "content": system_prompt}]
    
    if request.previous_messages:
        messages.extend(request.previous_messages[-10:])
        
    user_content = request.user_msg
    if request.rag_context:
        user_content = f"Context:\n{request.rag_context}\n\nQuestion: {request.user_msg}"
    
    messages.append({"role": "user", "content": user_content})

    async def event_generator():
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
             yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
