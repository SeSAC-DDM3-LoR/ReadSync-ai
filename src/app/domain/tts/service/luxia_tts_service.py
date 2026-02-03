import os
import json
import httpx
import asyncio
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class LuxiaTTSService:
    def __init__(self):
        self.api_key = os.getenv("LUXIA_API_KEY")
        self.api_url = "https://bridge.luxiacloud.com/luxia/v1/text-to-speech"
        
        # 화자별 목소리 ID 매핑 (필요에 따라 ID 변경 가능)
        # JSON 파일의 "speakers" 목록: ["나레이션", "까만 수염", "선원", ...]
        self.voice_map = {
            "나레이션": 76,  # 기본 차분한 목소리 (예시)
            "까만 수염": 76,  # 다른 ID가 있다면 변경 (예: 77)
            "선원": 76,
            "구경꾼": 76,
            "나": 76,
            "인버네스(형사)": 76
        }
        self.default_voice = 76

    async def _request_tts(self, text: str, voice_id: int) -> bytes:
        """Luxia API에 TTS 요청을 보내고 오디오 바이너리 데이터를 반환합니다."""
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "voice": voice_id,
            "lang": "ko"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=30.0 # 타임아웃 설정
                )
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as e:
                print(f"TTS API Error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                print(f"Request Failed: {str(e)}")
                return None

    async def convert_json_to_audio(self, json_path: str, output_dir: str):
        """
        JSON 파일을 읽어 각 문단을 오디오 파일로 변환하여 저장합니다.
        """
        # 1. JSON 파일 로드
        file_path = Path(json_path)
        if not file_path.exists():
            print(f"File not found: {json_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 출력 디렉토리 생성
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing book: {data.get('book_name', 'Unknown')}")
        
        # 3. 각 문단 처리 (비동기 처리 가능하지만, 순차 처리로 예시 작성)
        content_list = data.get("content", [])
        
        for item in content_list:
            item_id = item.get("id")
            text = item.get("text")
            speaker = item.get("speaker")
            
            # 텍스트가 없거나 너무 짧으면 스킵
            if not text or len(text.strip()) == 0:
                continue

            # 화자에 따른 보이스 ID 결정
            voice_id = self.voice_map.get(speaker, self.default_voice)
            
            print(f"Generating audio for [{item_id}] Speaker: {speaker}...")
            
            # API 호출
            audio_data = await self._request_tts(text, voice_id)
            
            # 파일 저장
            if audio_data:
                file_name = f"{item_id}.wav"
                save_path = out_path / file_name
                with open(save_path, "wb") as audio_file:
                    audio_file.write(audio_data)
                print(f"Saved: {save_path}")
            else:
                print(f"Failed to generate audio for {item_id}")

            # API 속도 제한 고려 시 약간의 딜레이 추가 가능
            # await asyncio.sleep(0.5)

# 실행 예시 (직접 실행 시)
if __name__ == "__main__":
    async def main():
        service = LuxiaTTSService()
        
        # 테스트할 JSON 파일 경로 (업로드된 파일 기준)
        json_file = "src/app/domain/tts/test_resources/book/만세전_chapter4.json"
        
        # 결과 저장 경로
        output_folder = "output/audio/만세전_chapter4"
        
        await service.convert_json_to_audio(json_file, output_folder)

    asyncio.run(main())