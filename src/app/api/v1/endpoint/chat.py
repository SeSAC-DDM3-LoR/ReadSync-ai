from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
from app.domain.tts.service.luxia_tts_service import AiChatService

router = APIRouter()
chat_service = AiChatService()

# DTOs
class AiClassifyRequest(BaseModel):
    user_msg: str
    chapter_id: Optional[int] = None
    current_paragraph_content: Optional[str] = None

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
