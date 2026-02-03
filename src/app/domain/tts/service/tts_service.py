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
        
        # 데모용: 실제로는 DB에서 가져와야 할 텍스트 데이터를 파일에서 로드한다고 가정
        self.json_path = "src/app/domain/tts/test_resources/book/만세전_chapter4.json"

    async def get_audio_url(self, paragraph_id: str, chapter_id: str = "ch4") -> str:
        """
        프론트엔드 요청을 처리하는 메인 함수
        """
        # S3 키 생성 (경로 구조: book/chapter/paragraph.wav)
        s3_key = f"books/만세전/{chapter_id}/{paragraph_id}.wav"

        # 1. S3에 파일이 있는지 확인
        if self._check_s3_exists(s3_key):
            print(f"Cache Hit! S3에서 반환: {s3_key}")
            return self._generate_presigned_url(s3_key)

        # 2. 파일이 없다면 (Cache Miss) -> 텍스트 찾기
        text = self._find_text_by_id(paragraph_id)
        if not text:
            raise ValueError("해당 ID의 텍스트를 찾을 수 없습니다.")

        # 3. Luxia API로 오디오 생성
        print(f"Cache Miss. Luxia API 호출 중...: {paragraph_id}")
        audio_data = await self._call_luxia_tts(text)

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

    async def _call_luxia_tts(self, text: str) -> bytes:
        """Luxia API 호출 (이전 코드 재사용)"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://bridge.luxiacloud.com/luxia/v1/text-to-speech",
                    headers={"apikey": self.luxia_api_key, "Content-Type": "application/json"},
                    json={"input": text, "voice": 76, "lang": "ko"},
                    timeout=30.0
                )
                response.raise_for_status()
                return response.content
            except Exception as e:
                print(f"TTS API Error: {e}")
                return None

    def _find_text_by_id(self, p_id: str) -> str:
        """JSON 파일에서 ID로 텍스트 검색 (DB 대용)"""
        # 실제 운영시에는 DB 조회를 권장합니다.
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data['content']:
                    if item['id'] == p_id:
                        return item['text']
        except Exception:
            return None
        return None