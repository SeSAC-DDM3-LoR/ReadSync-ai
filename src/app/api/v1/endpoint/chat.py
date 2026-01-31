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
    chat_type: str  # DEFINITION, CONTENT_QA, CONTENT_QA_CONTEXT, SUMMARY, QUIZ, CHIT_CHAT

class GenerateRequest(BaseModel):
    user_msg: str
    chat_type: str
    rag_context: Optional[str] = None
    previous_messages: Optional[List[dict]] = None # [{"role": "user", "content": "..."}, ...]

class GenerateResponse(BaseModel):
    response: str
    token_usage: Optional[int] = 0

class AiRewriteRequest(BaseModel):
    user_msg: str
    previous_messages: Optional[List[dict]] = None

class AiRewriteResponse(BaseModel):
    rewritten_query: str

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
    2. CONTENT_QA_CONTEXT: [PRIORITY CHECK] Asking questions that CONTAIN PRONOUNS (he, she, it, they, that, this) or refer to previous conversation context. If the question is "Why did he do that?" or "What happened next?", it MUST be classified as this.
    3. CONTENT_QA: Asking specific questions about the book's content where the user's question is COMPLETE and INDEPENDENT. No previous context is needed. (e.g., "Who is the protagonist?", "What happened in chapter 1?")
    4. SUMMARY: Asking to summarize the current chapter or section. (e.g., "Summarize this chapter", "What is this page about?")
    5. QUIZ: Asking to generate a quiz or test about the book. (e.g., "Give me a quiz", "Test my understanding")
    6. CHIT_CHAT: General conversation, greetings, or questions unrelated to the book content. (e.g., "Hello", "You are smart", "Thank you")
    
    Return ONLY the category name. Do not add any other text.
    """
    
    user_content = f"User Message: {request.user_msg}"
    if request.current_paragraph_content:
        user_content += f"\n(Context: User is reading this paragraph: '{request.current_paragraph_content[:100]}...')"

    try:
        # Responses API 사용
        response = client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input=user_content,
            instructions=system_prompt,
            reasoning={"effort": "low"},
            max_output_tokens=20
        )
        
        chat_type = response.output_text.strip().upper()
        
        # 유효성 검사 (혹시 모를 환각 방지)
        valid_types = ["DEFINITION", "CONTENT_QA", "CONTENT_QA_CONTEXT", "SUMMARY", "QUIZ", "CHIT_CHAT"]
        if chat_type not in valid_types:
            # 기본값으로 CONTENT_QA 또는 CHIT_CHAT 설정
            if "MEAN" in chat_type or "DEFINE" in chat_type:
                chat_type = "DEFINITION"
            elif "CONTEXT" in chat_type:
                 chat_type = "CONTENT_QA_CONTEXT"
            else:
                chat_type = "CONTENT_QA"
        
        # [Heuristic Fallback]
        # 모델이 CONTENT_QA로 분류했더라도, 대명사가 포함되어 있으면 CONTENT_QA_CONTEXT로 강제 조정
        if chat_type == "CONTENT_QA":
            # 간단한 키워드 검사 (영어/한국어)
            # 영어는 단어 경계(\b) 체크 필요
            eng_pronouns = ["he", "she", "it", "they", "that", "this", "him", "her", "his", "their"]
            kor_pronouns = ["그는", "그가", "그녀", "그건", "이건", "저건", "걔", "왜", "누구"] 
            
            lower_msg = request.user_msg.lower()
            import re
            
            found = False
            for p in eng_pronouns:
                if re.search(r'\b' + re.escape(p) + r'\b', lower_msg):
                    found = True
                    break
            
            if not found:
                for p in kor_pronouns:
                    if p in lower_msg: # 한국어는 조사가 붙으므로 substring 체크가 맞음
                        found = True
                        break
            
            if found:
                chat_type = "CONTENT_QA_CONTEXT"

        return ClassifyResponse(chat_type=chat_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/rewrite-query", response_model=AiRewriteResponse)
async def rewrite_query(request: AiRewriteRequest):
    """
    이전 대화 기록을 바탕으로 사용자 질문을 재작성합니다 (RAG 검색 품질 향상용).
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured")

    system_prompt = """
    You are a query rewriting assistant for a book RAG system.
    Based on the conversation history and the user's last message, rewrite the user's message into a standalone question that can be understood without context.
    
    Rules:
    1. Resolve any pronouns (he, she, it, that, they) using the history.
    2. Include specific names or events references from the history if relevant.
    3. If the user's message is already standalone, return it exactly as is.
    4. Return ONLY the rewritten query text.
    """
    
    # 히스토리 포맷팅
    history_text = ""
    if request.previous_messages:
        # 최근 5개만 사용
        for msg in request.previous_messages[-5:]:
             role = msg.get("role", "user")
             content = msg.get("content", "")
             history_text += f"{role}: {content}\n"
    
    input_text = f"History:\n{history_text}\nUser Message: {request.user_msg}\nRewritten Query:"

    try:
        response = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=input_text,
            instructions=system_prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        )
        
        rewritten = response.output_text.strip()
        # 따옴표 제거 등의 간단한 후처리
        rewritten = rewritten.replace('"', '').replace("'", "")
        
        return AiRewriteResponse(rewritten_query=rewritten)

    except Exception as e:
        # 에러 발생 시 원본 반환 (Fail-safe)
        print(f"Rewrite failed: {e}")
        return AiRewriteResponse(rewritten_query=request.user_msg)

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
        if request.chat_type in ["CONTENT_QA", "CONTENT_QA_CONTEXT", "SUMMARY", "QUIZ"]:
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
            if request.chat_type in ["CONTENT_QA", "CONTENT_QA_CONTEXT", "SUMMARY", "QUIZ"]:
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
