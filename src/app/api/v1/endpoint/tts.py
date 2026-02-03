from fastapi import APIRouter, HTTPException
from app.domain.tts.service.tts_service import TtsService

router = APIRouter()
tts_service = TtsService()

@router.get("/play/{paragraph_id}")
async def play_audio(paragraph_id: str):
    try:
        # 오디오 URL을 받아옴 (S3 URL)
        audio_url = await tts_service.get_audio_url(paragraph_id)
        return {"url": audio_url, "paragraph_id": paragraph_id}
    except ValueError:
        raise HTTPException(status_code=404, detail="텍스트를 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))