import boto3
import httpx
import json
from botocore.exceptions import ClientError
from pathlib import Path
from app.core.config import settings

class TtsService:
    def __init__(self):
        # AWS S3 Client 초기화
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME
        self.luxia_api_key = settings.LUXIA_API_KEY
        


    async def get_audio_url(self, paragraph_id: str, text: str, chapter_id: str = "ch4", voice_id: int = 76) -> str:
        """
        프론트엔드 요청을 처리하는 메인 함수
        
        Args:
            paragraph_id: 문단 ID (예: p_0001)
            text: TTS로 변환할 텍스트 내용
            chapter_id: 챕터 ID (예: ch4)
            voice_id: Luxia Voice ID (기본값: 76 - SEONBI)
        
        Returns:
            S3 Presigned URL
        """
        # S3 키 생성 (경로 구조: books/{chapter_id}/voice_{voice_id}/{paragraph_id}.wav)
        # voice_id를 포함하여 같은 paragraph라도 다른 voice면 다른 파일로 저장
        s3_key = f"books/{chapter_id}/voice_{voice_id}/{paragraph_id}.wav"

        # 1. S3에 파일이 있는지 확인
        if self._check_s3_exists(s3_key):
            print(f"Cache Hit! S3에서 반환: {s3_key}")
            return self._generate_presigned_url(s3_key)

        # 2. 텍스트 유효성 검사 (백엔드에서 전달받은 text 사용)
        if not text:
            raise ValueError("TTS 변환할 텍스트가 없습니다.")

        # 3. Luxia API로 오디오 생성 (voice_id 전달)
        print(f"Cache Miss. Luxia API 호출 중... paragraph: {paragraph_id}, voice: {voice_id}")
        audio_data = await self._call_luxia_tts(text, voice_id)

        if not audio_data:
            raise Exception("TTS 생성 실패")

        # 4. S3 업로드
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=audio_data,
            ContentType='audio/wav'
        )
        
        # 5. URL 반환
        return self._generate_presigned_url(s3_key)

    def _check_s3_exists(self, key: str) -> bool:
        """S3에 객체가 존재하는지 확인"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def _generate_presigned_url(self, key: str) -> str:
        """프론트엔드가 바로 재생할 수 있는 임시 URL 생성 (유효기간 1시간)"""
        return self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': key},
            ExpiresIn=3600
        )

    async def _call_luxia_tts(self, text: str, voice_id: int = 76) -> bytes:
        """
        Luxia API 호출
        
        Args:
            text: TTS로 변환할 텍스트
            voice_id: Luxia Voice ID (기본값: 76 - SEONBI)
        
        Returns:
            오디오 데이터 (bytes)
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://bridge.luxiacloud.com/luxia/v1/text-to-speech",
                    headers={"apikey": self.luxia_api_key, "Content-Type": "application/json"},
                    json={"input": text, "voice": voice_id, "lang": "ko"},
                    timeout=30.0
                )
                response.raise_for_status()
                print(f"Luxia TTS 성공: text length={len(text)}, voice={voice_id}")
                return response.content
            except Exception as e:
                print(f"TTS API Error: {e}")
                return None

