from app.domain.tts.service.tts_service import TtsService
from pydantic import BaseModel

router = APIRouter()
tts_service = TtsService()

class TtsRequest(BaseModel):
    voice_id: int = 76
    text: str

@router.post("/play/{paragraph_id}")
async def play_audio(
    paragraph_id: str,
    request: TtsRequest
):
    """
    TTS 오디오 URL 반환 (POST)
    
    - **paragraph_id**: 문단 ID (예: p_0001)
    - **voice_id**: Luxia Voice ID (76: SEONBI, 2: BORAM, 5: YUNA, 7: KYEON, 8: BITNA)
    - **text**: TTS로 변환할 텍스트
    """
    try:
        # 오디오 URL을 받아옴 (S3 URL) - voice_id & text 전달
        audio_url = await tts_service.get_audio_url(paragraph_id, text=request.text, voice_id=request.voice_id)
        return {"url": audio_url, "paragraph_id": paragraph_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="입력값이 올바르지 않습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))