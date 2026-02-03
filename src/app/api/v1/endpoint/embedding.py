# 모델 로드하는 경우(hugging face api가 아닌 로컬 다운)
# from fastapi import APIRouter, HTTPException, status
# from pydantic import BaseModel, Field
# from app.lifespan import ml_models
# import logging

# # 로그 기록 설정
# logger = logging.getLogger(__name__)
# router = APIRouter()

# # 1. 자바에서 보낼 데이터 규격 (유효성 검사 포함)
# class EmbeddingRequest(BaseModel):
#     text: str = Field(..., min_length=1, description="임베딩할 도서의 제목이나 줄거리")

# # 2. 자바에 돌려줄 응답 규격
# class EmbeddingResponse(BaseModel):
#     embedding: list[float] = Field(..., description="KURE-v1 모델이 생성한 1024차원 벡터")

# @router.post(
#     "/embed", 
#     response_model=EmbeddingResponse,
#     status_code=status.HTTP_200_OK,
#     summary="실시간 단일 텍스트 임베딩 생성"
# )
# async def get_embedding(request: EmbeddingRequest):
#     """
#     자바 서버로부터 받은 텍스트를 KURE-v1 모델을 사용하여 벡터로 변환합니다.
#     """
#     # [체크] 모델 로드 여부 확인
#     if "embedding_model" not in ml_models:
#         logger.error("AI Model (KURE-v1) is not loaded in ml_models.")
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
#             detail="AI 모델이 아직 준비되지 않았습니다. 서버 로그를 확인하세요."
#         )

#     try:
#         # [실행] 임베딩 생성
#         model = ml_models["embedding_model"]
        
#         # SentenceTransformer의 encode는 기본적으로 CPU/GPU 자원을 사용하므로 
#         # 단일 텍스트 처리 시 매우 빠릅니다.
#         # .tolist()를 호출하여 JSON 응답이 가능한 파이썬 리스트로 변환합니다.
#         vector = model.encode(request.text).tolist()
        
#         logger.info(f"Successfully generated embedding for text: {request.text[:20]}...")
#         return EmbeddingResponse(embedding=vector)

#     except Exception as e:
#         logger.error(f"Error during embedding generation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"임베딩 생성 중 서버 내부 오류가 발생했습니다: {str(e)}"
#         )

import os
import json
import numpy as np
import aioboto3
import asyncio
from fastapi import APIRouter, HTTPException
from typing import List
from urllib.parse import urlparse, unquote
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv


import re
import requests

# .env 로드 및 설정
load_dotenv()
router = APIRouter()
TOKEN_LIMIT = 2000 # 청크 분절 기준

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)

# [Global Limiter] 전체 애플리케이션 수준에서 동시 요청 수 제한
# 함수 내부가 아닌 전역 변수로 선언해야 여러 책을 동시에 처리할 때도 총합을 제한할 수 있습니다.
GLOBAL_SEMAPHORE = asyncio.Semaphore(20) # 5 -> 20으로 상향 (안정성 확인 필요)

# 1. Google Drive 다운로드 함수
async def download_from_drive(google_drive_url: str):
    try:
        # 파일 ID 추출
        file_id_match = re.search(r'/d/([^/]+)', google_drive_url)
        if not file_id_match:
            raise HTTPException(status_code=400, detail="유효하지 않은 구글 드라이브 링크입니다.")
        
        file_id = file_id_match.group(1)
        
        # 2. 직속 다운로드 URL 생성
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # 3. 파일 다운로드 (Non-blocking)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, requests.get, download_url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="구글 드라이브 파일을 가져오지 못했습니다. 공유 설정을 확인하세요.")
            
        return response.json()
    except Exception as e:
        print(f"❌ Drive 다운로드 실패: {e}")
        raise e

# 2. S3 다운로드 함수
async def download_from_s3(s3_url: str):
    # 🔍 1. 디버깅: 환경 변수가 제대로 들어왔는지 로그로 확인
    ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
    SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
    REGION = os.getenv("AWS_REGION", "ap-northeast-2")
    # BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "")
    
    if not ACCESS_KEY or not SECRET_KEY:
        print("❌ AWS 자격 증명(환경 변수)이 없습니다! docker-compose.yml을 확인하세요.")
    else:
        print(f"🔑 AWS Key 로드 성공: {ACCESS_KEY[:4]}****")

    try:
        # 🔍 2. URL 파싱 로직 (s3:// 프로토콜과 https:// URL 모두 대응하도록 보완)
        parsed_url = urlparse(s3_url)
        
        # 's3://버킷명/키' 형식인 경우
        if parsed_url.scheme == 's3':
            bucket_name = parsed_url.netloc
            key = unquote(parsed_url.path.lstrip('/'))
        # 'https://버킷명.s3...' 형식인 경우
        else:
            bucket_name = parsed_url.netloc.split('.')[0]
            key = unquote(parsed_url.path.lstrip('/'))

        # 🔍 3. 세션 생성 시 명시적으로 자격 증명 주입 (가장 안전함)
        session = aioboto3.Session(
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=REGION
        )

        async with session.client('s3') as s3:
            print(f"⬇️ 다운로드 시작: {bucket_name}/{key}")
            response = await s3.get_object(Bucket=bucket_name, Key=key)
            async with response['Body'] as stream:
                file_content = await stream.read()
                return json.loads(file_content.decode('utf-8'))

    except Exception as e:
        print(f"❌ S3 다운로드 에러 상세: {str(e)}")
        # 에러를 감추지 말고 호출한 쪽(FastAPI)에서 500 에러 원인을 알 수 있게 던짐
        raise e

def aggregate_vectors(vectors):
    """
    여러 텍스트 청크의 벡터를 하나로 합칩니다.
    [Hybrid Aggregation] Mean Pooling + Max Pooling
    """
    if not vectors:
        return None
    
    
    arr = np.array(vectors)
    
    # [Main] Power Mean Aggregation (p=3)
    # 산술 평균(p=1)과 Max Pooling(p=무한대)의 절충안.
    # 각 차원별 값의 크기를 p제곱하여 평균을 냄으로써,
    # 강하게 발현된 특징(Keyword/Theme)을 "적당히 강조"하고(Sharpening), 
    # 너무 약한 신호(Noise)는 억제합니다.
    try:
        # 0. 사전 정규화 (필수)
        # Power Mean을 쓰려면 각 벡터의 스케일이 맞춰져 있어야 합니다.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr_normalized = arr / (norms + 1e-9)

        # 1. Power Mean (p=2.5)
        # p=3이 너무 튈 수 있다는 우려를 반영하여 2.5로 하향 조정.
        # 여전히 특징은 잘 살리지만(Sharpening), 3.0보다는 부드럽고 안정적입니다.
        p = 2.5
        power_arr = np.sign(arr_normalized) * np.power(np.abs(arr_normalized), p)
        mean_vec = np.mean(power_arr, axis=0)
        
        # 2. 다시 원래 스케일로 복원 (Inverse Power)
        # integrated_vec = sign(mean) * |mean|^(1/p)
        integrated_vec = np.sign(mean_vec) * np.power(np.abs(mean_vec), 1.0/p)
        
    except Exception as e:
        print(f"⚠️ Aggregation Failed, using simple mean: {e}")
        integrated_vec = np.mean(arr, axis=0)

    # 최종 정규화 (L2 Norm)
    norm = np.linalg.norm(integrated_vec)
    if norm > 1e-9:
        integrated_vec = integrated_vec / norm
        
    return integrated_vec
    
async def core_embedding_logic(path: str):
    if "drive.google.com" in path:
        book_data = await download_from_drive(path)
    elif "amazonaws.com" in path or ".s3." in path:
        book_data = await download_from_s3(path)
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 경로 형식입니다.")

    texts = [node['text'] for node in book_data.get('content', []) if 'text' in node]
    
    # 1. 먼저 텍스트를 청크로 모두 분할 (메모리 작업)
    chunks_to_embed = []
    current_chunk = ""
    
    for text in texts:
        if len(current_chunk) + len(text) > TOKEN_LIMIT:
            if current_chunk.strip():
                chunks_to_embed.append(current_chunk.strip())
                current_chunk = ""

            # 1-2. [중요] 새로 들어온 text 자체가 limit보다 크다면? 
            if len(text) > TOKEN_LIMIT:
                sub_chunks = [text[i : i + TOKEN_LIMIT] for i in range(0, len(text), TOKEN_LIMIT)]
                # 마지막 조각은 다음 text와 합치기 위해 남겨두고 나머지는 즉시 리스트에 추가
                for sub in sub_chunks[:-1]:
                    if sub.strip():
                        chunks_to_embed.append(sub.strip())
                current_chunk = sub_chunks[-1] # 마지막 조각만 유지
            else:
                current_chunk = text

        else:
            current_chunk += " " + text
    
    # 마지막 남은 청크 처리
    if current_chunk.strip():
        chunks_to_embed.append(current_chunk.strip())

    if not chunks_to_embed:
        raise HTTPException(status_code=400, detail="임베딩할 텍스트 내용이 없습니다.")

    # 2. [병렬 처리] 모아둔 청크를 한꺼번에 임베딩 요청
    # 기존: asyncio.gather로 무제한 요청 -> 504 Gateway Timeout 발생
    # 변경: Global Semaphore로 전체 동시 요청 수 제한 + Retry 로직 적용

    async def safe_embedding_request(text_chunk):
        async with GLOBAL_SEMAPHORE:
            max_retries = 5 # 재시도 횟수 증가
            base_delay = 0.5 # 초기 대기 시간 단축 (2s -> 0.5s)
            
            for attempt in range(max_retries):
                try:
                    return await client.feature_extraction(text_chunk)
                except Exception as e:
                    # 504(Gateway Timeout), 502(Bad Gateway), 429(Too Many Requests) 등은 재시도 가치 있음
                    error_msg = str(e)
                    if "504" in error_msg or "502" in error_msg or "429" in error_msg:
                        if attempt < max_retries - 1:
                            # Exponential Backoff with Jitter (Optional)
                            wait_time = base_delay * (2 ** attempt) 
                            # print(f"⚠️ API Error ({e}), retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    # 재시도 불가능한 에러이거나 횟수 초과 시
                    print(f"❌ Feature Extraction Failed after {attempt+1} attempts: {e}")
                    raise e

    try:
        tasks = [safe_embedding_request(chunk) for chunk in chunks_to_embed]
        embedding_list = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"❌ Parallel Embedding Error: {e}")
        raise e

    # if not chunks:
    if not embedding_list:
        raise HTTPException(status_code=400, detail="임베딩할 텍스트 내용이 없습니다.")

    # integrated_vector = np.mean(embedding_list, axis=0)
    integrated_vector = aggregate_vectors(embedding_list)

    result_list = integrated_vector.tolist() if hasattr(integrated_vector, "tolist") else integrated_vector

    print(f"총 임베딩된 청크 수: {len(embedding_list)}")

    return result_list

class S3EmbeddingRequest(BaseModel):
    s3_url: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

@router.post("/embed-from-s3", response_model=EmbeddingResponse)
async def get_embedding(request: S3EmbeddingRequest):
    vector = await core_embedding_logic(request.s3_url)
    return {"embedding": vector}

    # except Exception as e:
    #     print(f"❌ S3 비동기 임베딩 처리 실패: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))


class DriveEmbeddingRequest(BaseModel):
    google_drive_url: str

@router.post("/embed-from-drive")
async def get_embedding_from_drive(request: DriveEmbeddingRequest):
    vector = await core_embedding_logic(request.google_drive_url)
    return {"embedding": vector.tolist()}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
    

@router.post("/text-from-drive")
async def get_text_from_drive(request: DriveEmbeddingRequest) -> List[str]:
    # 1. 구글 드라이브 공유 링크 -> 직속 다운로드 링크로 변환
    # 링크 예시: https://drive.google.com/file/d/1A2B3C.../view?usp=sharing
    file_id_match = re.search(r'/d/([^/]+)', request.google_drive_url)
    if not file_id_match:
        raise HTTPException(status_code=400, detail="유효하지 않은 구글 드라이브 링크입니다.")
    
    file_id = file_id_match.group(1)
    
    # 2. 직속 다운로드 URL 생성
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    # 3. 파일 다운로드 (Non-blocking)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, requests.get, download_url)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="구글 드라이브 파일을 가져오지 못했습니다. 공유 설정을 확인하세요.")
        
    book_data = response.json()


    # 3. 데이터 구조에서 'text' 필드만 추출 (정우님의 JSON 규격 기준)
    # book_data['content'] 리스트를 돌며 'text' 키가 있는 것만 수집합니다.
    texts = [
        node['text'] 
        for node in book_data.get('content', []) 
        if 'text' in node
    ]

    return texts

class TextEmbeddingRequest(BaseModel):
    text: str

@router.post("/embed-text")
async def get_embedding_from_text(request: TextEmbeddingRequest):
    try:
        # 허깅페이스 API 호출 (await 사용)
        embedding = await client.feature_extraction(request.text)
        
        # 반환된 결과가 리스트 형태인지 확인 후 전달
        # 보통 feature_extraction은 리스트나 넘파이 배열 형태를 반환합니다.
        return {"embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace API 오류: {str(e)}")

class RagEmbeddingRequest(BaseModel):
    content: List[dict] # 책의 content 구조 (text, id 포함)


class RagNode(BaseModel):
    text: str
    id: str | int | None  # id can be int or string, normalize to string later
    speaker: str | None = None

class RagEmbeddingRequest(BaseModel):
    content: List[RagNode]

class ChildChunk(BaseModel):
    content_text: str  # Renamed
    vector: List[float]
    chunk_index: int   # Added
    paragraph_ids: List[str] # Changed to list of strings

class ParentChunk(BaseModel):
    content_text: str # Renamed
    speaker_list: List[str]
    paragraph_ids: List[str] # Added
    start_paragraph_id: str # Changed to str
    end_paragraph_id: str   # Changed to str
    children: List[ChildChunk]

class EmbeddingRagResponse(BaseModel):
    parents: List[ParentChunk]

class EmbeddingQueryRequest(BaseModel):
    text: str

@router.post("/embed-query", response_model=EmbeddingResponse)
async def embed_query(request: EmbeddingQueryRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty.")

        # 단일 텍스트 임베딩
        vector = await client.feature_extraction(request.text.strip())
        
        # 안전한 타입 변환
        result_list = vector.tolist() if hasattr(vector, "tolist") else vector
        return {"embedding": result_list}

    except Exception as e:
        print(f"❌ Query 임베딩 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed-rag-content", response_model=EmbeddingRagResponse)
async def embed_rag_content(request: RagEmbeddingRequest):
    try:
        # 1. 텍스트 추출 및 구조화 (Pydantic 모델 사용)
        content_nodes = request.content
        if not content_nodes:
            raise HTTPException(status_code=400, detail="임베딩할 텍스트 내용이 없습니다.")

        parents = []
        
        # Parent Chunking 설정
        PARENT_CHUNK_SIZE = 20  # 문단 수 기준
        PARENT_OVERLAP = 5      # 20% 오버랩
        
        # Child Chunking 설정
        CHILD_CHUNK_SIZE = 5    # 5문단
        CHILD_OVERLAP = 1       # 1문단 오버랩

        # 전체 노드 순회
        parent_start_idx = 0
        
        while parent_start_idx < len(content_nodes):
            # 2-1. Parent Chunk 범위 설정
            parent_end_idx = min(parent_start_idx + PARENT_CHUNK_SIZE, len(content_nodes))
            
            # Parent Chunk 생성
            parent_nodes = content_nodes[parent_start_idx:parent_end_idx]
            
            # Parent 메타데이터 추출
            parent_text_builder = []
            parent_speakers = set()
            parent_para_ids = [] # Added for ID collection
            start_para_id = str(parent_nodes[0].id) if parent_nodes[0].id is not None else "0"
            end_para_id = str(parent_nodes[-1].id) if parent_nodes[-1].id is not None else "0"
            
            for node in parent_nodes:
                text_part = node.text
                if node.speaker:
                    text_part = f"{node.speaker}: {text_part}"
                    parent_speakers.add(node.speaker)
                parent_text_builder.append(text_part)
                parent_para_ids.append(str(node.id) if node.id is not None else "0") # Collect IDs
            
            parent_content = " ".join(parent_text_builder)
            
            # 2-2. Child Chunking (Parent 내부에서 수행)
            children = []
            child_start_idx = 0 # Parent 내부 인덱스
            
            while child_start_idx < len(parent_nodes):
                child_end_idx = min(child_start_idx + CHILD_CHUNK_SIZE, len(parent_nodes))
                child_nodes = parent_nodes[child_start_idx:child_end_idx]
                
                # Child 메타데이터
                child_text_builder = []
                child_para_ids = []

                for node in child_nodes:
                    text_part = node.text
                    if node.speaker:
                        text_part = f"{node.speaker}: {text_part}"
                    child_text_builder.append(text_part)
                    child_para_ids.append(str(node.id) if node.id is not None else "0")
                
                child_content = " ".join(child_text_builder)
                
                # Child Vector 생성 (비동기 처리)
                if child_content.strip():
                     vector = await client.feature_extraction(child_content.strip())
                     
                     children.append(ChildChunk(
                         content_text=child_content,
                         vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                         chunk_index=len(children), # 현재 Parent 내에서의 순서 (0부터 시작)
                         paragraph_ids=child_para_ids
                     ))
                
                # Child Loop Control
                if child_end_idx == len(parent_nodes):
                    break
                
                # 인덱스 증가
                child_start_idx += (CHILD_CHUNK_SIZE - CHILD_OVERLAP)
            
            # Parent 결과 저장
            parents.append(ParentChunk(
                content_text=parent_content,
                speaker_list=list(parent_speakers),
                paragraph_ids=parent_para_ids, # Added
                start_paragraph_id=start_para_id,
                end_paragraph_id=end_para_id,
                children=children
            ))

            # Parent Loop Control
            if parent_end_idx == len(content_nodes):
                break
                
            parent_start_idx += (PARENT_CHUNK_SIZE - PARENT_OVERLAP)

        print(f"✅ RAG Parent 청킹 완료: 총 {len(parents)}개 Parent 청크 생성")
        return EmbeddingRagResponse(parents=parents)

    except Exception as e:
        print(f"❌ RAG 임베딩 처리 실패: {e}")
        # traceback 출력으로 디버깅 용이하게
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- [새로 추가할 배치 엔드포인트] ---
import asyncio

# --- [새로 추가할 배치 엔드포인트] ---
@router.post("/embed-batch")
async def get_batch_embeddings(request: dict):
    paths = request.get("paths", [])
    chapter_vectors = []

    # [수정] asyncio.gather를 사용하여 병렬 처리
    # 기존: 순차적 await -> 느림
    # 변경: 동시에 여러 요청 처리 -> 빠름
    
    # 1. 태스크 생성
    tasks = [core_embedding_logic(path) for path in paths]
    
    # 2. 병렬 실행
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 3. 결과 수집 (에러가 난 항목은 제외하거나 로깅)
    for res in results:
        if isinstance(res, Exception):
            print(f"⚠️ Batch processing error: {res}")
        elif res is not None:
            chapter_vectors.append(res)

    if not chapter_vectors:
        raise HTTPException(status_code=400, detail="임베딩할 수 있는 데이터가 없습니다.")

    # 북 벡터 계산 (모든 챕터 벡터의 평균)
    # average_vector = np.mean(chapter_vectors, axis=0)
    average_vector = aggregate_vectors(chapter_vectors)

    book_vector = average_vector.tolist() if hasattr(average_vector, "tolist") else average_vector

    return {
        "book_vector": book_vector,
        "chapter_vectors": [cv.tolist() if hasattr(cv, "tolist") else cv for cv in chapter_vectors]
    }