from fastapi import APIRouter, HTTPException, Query
from app.domain.tts.service.tts_service import TtsService

router = APIRouter()
tts_service = TtsService()

@router.get("/play/{paragraph_id}")
async def play_audio(
    paragraph_id: str,
    voice_id: int = Query(default=76, description="Luxia Voice ID (기본값: 76 - SEONBI)")
):
    """
    TTS 오디오 URL 반환
    
    - **paragraph_id**: 문단 ID (예: p_0001)
    - **voice_id**: Luxia Voice ID (76: SEONBI, 2: BORAM, 5: YUNA, 7: KYEON, 8: BITNA)
    """
    try:
        # 오디오 URL을 받아옴 (S3 URL) - voice_id 전달
        audio_url = await tts_service.get_audio_url(paragraph_id, voice_id=voice_id)
        return {"url": audio_url, "paragraph_id": paragraph_id}
    except ValueError:
        raise HTTPException(status_code=404, detail="텍스트를 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))