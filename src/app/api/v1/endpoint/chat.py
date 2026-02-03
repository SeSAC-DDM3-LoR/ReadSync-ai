<<<<<<< HEAD
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
from app.domain.tts.service.luxia_tts_service import AiChatService

router = APIRouter()
chat_service = AiChatService()

# DTOs
class AiClassifyRequest(BaseModel):
=======
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
>>>>>>> main
    user_msg: str
    chapter_id: Optional[int] = None
    current_paragraph_content: Optional[str] = None

<<<<<<< HEAD
class AiClassifyResponse(BaseModel):
    chat_type: str

class AiGenerateRequest(BaseModel):
    user_msg: str
    chat_type: str
    rag_context: Optional[str] = None

class AiGenerateResponse(BaseModel):
    response: str
    token_usage: Optional[int] = 0

@router.post("/chat/classify", response_model=AiClassifyResponse)
async def classify_chat(request: AiClassifyRequest):
    chat_type = chat_service.classify_intent(
        request.user_msg, 
        request.chapter_id, 
        request.current_paragraph_content
    )
    return AiClassifyResponse(chat_type=chat_type)

@router.post("/chat/generate", response_model=AiGenerateResponse)
async def generate_chat(request: AiGenerateRequest):
    answer = chat_service.generate_answer(
        request.user_msg, 
        request.chat_type, 
        request.rag_context
    )
    # Estimate token usage simply (or parse from complete response if updated)
    token_usage = len(answer) // 4 
    return AiGenerateResponse(response=answer, token_usage=token_usage)

@router.post("/chat/generate/stream")
async def generate_chat_stream(request: AiGenerateRequest):
    return StreamingResponse(
        chat_service.generate_answer_stream(
            request.user_msg, 
            request.chat_type, 
            request.rag_context
        ),
        media_type="text/event-stream"
    )
=======
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

def format_chat_history(previous_messages: Optional[List[dict]], current_msg: str, rag_context: Optional[str] = None) -> str:
    """
    대화 내역과 현재 메시지를 하나의 문자열로 포맷팅합니다. (Responses API의 input용)
    """
    formatted_input = ""
    
    # 이전 대화 내역 추가
    if previous_messages:
        # 최근 10개 메시지만 사용 (백엔드에서 이미 처리되지만 안전장치)
        recent_messages = previous_messages[-10:]
        for msg in recent_messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted_input += f"[{role}]: {content}\n\n"
    
    # RAG Context 추가
    if rag_context:
        formatted_input += f"[CONTEXT]\n{rag_context}\n\n"
        
    # 현재 사용자 메시지 추가
    formatted_input += f"[USER]: {current_msg}"
    
    return formatted_input

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
        # Responses API 사용
        response = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=user_content,
            instructions=system_prompt,
            reasoning={"effort": "minimal"},
            max_output_tokens=20
        )
        
        chat_type = response.output_text.strip().upper()
        
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
        system_prompt += " Answer the question based ONLY on the provided Context. If the answer is not in the Context, say '제공된 내용에서는 답을 찾을 수 없습니다.'"
    elif request.chat_type == "SUMMARY":
        system_prompt += " Summarize the provided text concisely."
    elif request.chat_type == "QUIZ":
        system_prompt += " Generate a multiple-choice quiz based on the context. Provide the question, 4 options, and the answer."
    elif request.chat_type == "CHIT_CHAT":
        system_prompt += " Engage in a friendly, polite conversation. Do not make up facts about the book if you don't know."

    # Input 문자열 구성
    input_text = format_chat_history(request.previous_messages, request.user_msg, request.rag_context)

    try:
        # Chat Type에 따른 모델 및 파라미터 설정
        if request.chat_type in ["CONTENT_QA", "SUMMARY", "QUIZ"]:
            # 추론이 필요한 작업: gpt-5-mini
            model = "gpt-5-mini-2025-08-07"
            reasoning_config = {"effort": "low"}
            text_config = {"verbosity": "medium"}
        else:
            # 가벼운 작업: gpt-5-nano
            model = "gpt-5-nano-2025-08-07"
            reasoning_config = {"effort": "minimal"}
            text_config = {"verbosity": "low"}

        response = client.responses.create(
            model=model,
            input=input_text,
            instructions=system_prompt,
            reasoning=reasoning_config,
            text=text_config
        )
        
        response_text = response.output_text
        # Responses API에서는 usage 정보가 다를 수 있음. 일단 0으로 처리하거나 response 객체 확인 필요.
        # 문서 상으로는 token_usage에 대한 명시가 없으나, 보통 usage 필드가 있음.
        token_usage = response.usage.total_tokens if response.usage else 0
        
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
        system_prompt += " Answer the question based ONLY on the provided Context. If the answer is not in the Context, say '제공된 내용에서는 답을 찾을 수 없습니다.'"
    elif request.chat_type == "SUMMARY":
        system_prompt += " Summarize the provided text concisely."
    elif request.chat_type == "QUIZ":
        system_prompt += " Generate a multiple-choice quiz based on the context. Provide the question, 4 options, and the answer."
    elif request.chat_type == "CHIT_CHAT":
        system_prompt += " Engage in a friendly, polite conversation. Do not make up facts about the book if you don't know."

    # Input 문자열 구성
    input_text = format_chat_history(request.previous_messages, request.user_msg, request.rag_context)

    async def event_generator():
        try:
             # Chat Type에 따른 모델 및 파라미터 설정
            if request.chat_type in ["CONTENT_QA", "SUMMARY", "QUIZ"]:
                # 추론이 필요한 작업: gpt-5-mini
                model = "gpt-5-mini-2025-08-07"
                reasoning_config = {"effort": "low"}
                text_config = {"verbosity": "medium"}
            else:
                # 가벼운 작업: gpt-5-nano
                model = "gpt-5-nano-2025-08-07"
                reasoning_config = {"effort": "minimal"}
                text_config = {"verbosity": "low"}

            stream = client.responses.create(
                model=model,
                input=input_text,
                instructions=system_prompt,
                reasoning=reasoning_config,
                text=text_config,
                stream=True
            )
            
            for event in stream:
                # ResponseTextDeltaEvent 처리
                if event.type == "response.output_text.delta":
                    # event.delta 가 텍스트 조각임
                    if event.delta:
                        yield event.delta
                        
        except Exception as e:
             yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
>>>>>>> main
