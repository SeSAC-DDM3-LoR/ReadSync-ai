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

class RewriteRequest(BaseModel):
    user_msg: str
    previous_messages: Optional[List[dict]] = None

class RewriteResponse(BaseModel):
    rewritten_msg: str

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
    Classify the user's message into one of the following categories based on the user's intent.

    Categories:
    1. DEFINITION: Asking for the meaning of a word, term, or concept.
       - Examples: 
         - "이 단어 뜻이 뭐야?"
         - "'ephemeral' 의미"
         - "설명이 필요해"
         - "'회귀'의 사전적 정의가 뭐야?"
         - "여기서 '객쩍다'는 무슨 뜻으로 쓰였어?"
         - "'만세'가 무슨 의미야?"
         - "이 단어 좀 쉽게 풀어서 설명해줘."
         - "'모던 보이'가 무슨 뜻이니?"

    2. CONTENT_QA_CONTEXT: The user is asking a follow-up question that relies on previous context (pronouns, implicit subject).
       - Examples:
         - "그는 왜 그랬어?" (Who is 'he'?)
         - "서울로 가서 뭘 했어?" (Missing subject, likely refers to previous topic)
         - "그 다음은 어떻게 됐어?"
         - "왜 죽었어?" (Who died?)
         - "그게 무슨 의미야?"
         - "그는 왜 그렇게 말했어?"
         - "그 여자는 결국 어떻게 됐어?"
         - "거기서 누구를 만났니?"
         - "왜 하필 그때 떠난 거야?"
         - "그 돈은 어디서 났어?"
         - "그들의 관계는 어때?"
       - Keywords: "그는", "그녀는", "그들은", "거기서", "그래서", "왜", "누구랑" (when used without explicit subject)

    3. CONTENT_QA: Specific questions about the book's content that are self-contained and explicitly name the subject.
       - Examples:
         - "김천 형님은 누구야?"
         - "만세전의 주인공이 서울로 간 이유는?"
         - "챕터 1에서 무슨 일이 있었어?"
         - "이인화는 왜 아내를 좋아하지 않아?"
         - "정자는 결국 누구와 결혼했니?"
         - "만세전의 결말은 어떻게 끝나?"
         - "주인공이 동경에서 받은 전보 내용은 뭐야?"
         - "김천 형님이 운영하는 가게는 뭐야?"

    4. SUMMARY: Requests to summarize the current chapter or section.
       - Examples: "요약해줘", "줄거리 알려줘", "3줄 요약"

    5. QUIZ: Requests for a quiz or test.
       - Examples: "퀴즈 내줘", "문제 풀어볼래"

    6. CHIT_CHAT: General conversation, greetings, or questions unrelated to the book content.
       - Examples: "안녕", "반가워", "고마워", "너 똑똑하다"

    IMPORTANT: If the question contains pronouns (그, 그녀, 걔) or lacks a specific subject (implying context dependence), classify as CONTENT_QA_CONTEXT.

    Response Constraints:
    - Return ONLY the category name.
    - NO punctuation, NO explanations, NO intro text.
    - Correct response example : "CONTENT_QA_CONTEXT"
    - Incorrect response example : "The category is CONTENT_QA_CONTEXT", "CONTENT_QA_CONTEXT.", "Category: CONTENT_QA_CONTEXT"
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
        valid_types = ["DEFINITION", "CONTENT_QA", "CONTENT_QA_CONTEXT", "SUMMARY", "QUIZ", "CHIT_CHAT"]
        if chat_type not in valid_types:
            # Fallback for messy output (e.g., "Category: CONTENT_QA_CONTEXT")
            if "CONTENT_QA_CONTEXT" in chat_type:
                chat_type = "CONTENT_QA_CONTEXT"
            elif "CONTENT_QA" in chat_type:
                chat_type = "CONTENT_QA"
            elif "MEAN" in chat_type or "DEFINE" in chat_type:
                chat_type = "DEFINITION"
            elif "KNOW" in chat_type or "QUIZ" in chat_type:
                chat_type = "QUIZ"
            elif "SUM" in chat_type:
                chat_type = "SUMMARY"
            else:
                chat_type = "CONTENT_QA" # Ultimate fallback
                
        return ClassifyResponse(chat_type=chat_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/context-rewrite", response_model=RewriteResponse)
async def rewrite_message(request: RewriteRequest):
    """
    이전 대화 문맥을 바탕으로 사용자의 질문을 명확하게 재작성합니다.
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured")

    system_prompt = """
    You are an AI assistant that rewrites user questions to be self-contained based on previous conversation history.
    
    1. Replace pronouns (he, she, it, they, etc.) with specific names or entities from the history.
    2. Resolve ambiguous references (e.g., "that event", "the previous chapter") using context.
    3. If the question is already clear, output it as is.
    4. Return ONLY the rewritten question. Do not add any explanation or labels.
    5. Language: Korean (Keep the original language style but make it specific).
    """
    
    formatted_history = ""
    if request.previous_messages:
        # 최근 5개 정도만 사용하여 문맥 파악
        for msg in request.previous_messages[-5:]:
             formatted_history += f"[{msg.get('role', 'user').upper()}]: {msg.get('content', '')}\n"
    
    user_input = f"{formatted_history}\n[USER]: {request.user_msg}\n\nRewritten Question:"

    try:
        response = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=user_input,
            instructions=system_prompt,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"}
        )
        
        rewritten_msg = response.output_text.strip()
        return RewriteResponse(rewritten_msg=rewritten_msg)

    except Exception as e:
        # 에러 발생 시 원래 메시지 반환 (Fallback)
        print(f"Rewrite failed: {e}")
        return RewriteResponse(rewritten_msg=request.user_msg)

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
