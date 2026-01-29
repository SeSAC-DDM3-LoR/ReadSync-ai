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
from fastapi import APIRouter, HTTPException
from typing import List
from urllib.parse import urlparse
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv


import re
import requests

# .env 로드 및 설정
load_dotenv()
router = APIRouter()
TOKEN_LIMIT = 3000 # 청크 분절 기준

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)

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
        
        # 3. 파일 다운로드
        response = requests.get(download_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="구글 드라이브 파일을 가져오지 못했습니다. 공유 설정을 확인하세요.")
            
        return response.json()
    except Exception as e:
        print(f"❌ Drive 다운로드 실패: {e}")
        raise e

# 2. S3 다운로드 함수
async def download_from_s3(s3_url: str):
    try:
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc.split('.')[0]
        key = parsed_url.path.lstrip('/')

        session = aioboto3.Session()
        async with session.client('s3') as s3:
            response = await s3.get_object(Bucket=bucket_name, Key=key)
            async with response['Body'] as stream:
                file_content = await stream.read()
                return json.loads(file_content.decode('utf-8'))
    except Exception as e:
        print(f"❌ S3 다운로드 실패: {e}")
        raise e

def aggregate_vectors(vectors, p=1.05, use_softmax=True):
    """
    여러 벡터를 비선형 가중치(Power Pooling + Softmax)를 적용하여 하나의 벡터로 집계합니다.
    p: 지수가 높을수록 '강한 특징'을 가진 벡터의 신호가 증폭됩니다.
    """
    if not vectors:
        return None
    
    arr = np.array(vectors)
    
    # 1. 벡터별 가중치 계산 (Softmax Weighting)
    if use_softmax:
        # 벡터의 L2 Norm이 큰(정보량이 많은) 벡터에 더 높은 가중치 부여
        norms = np.linalg.norm(arr, axis=1)
        exp_norms = np.exp(norms - np.max(norms))
        weights = exp_norms / np.sum(exp_norms)
        # 가중치 적용
        arr = arr * weights[:, np.newaxis]
    
    # 2. 원소별 신호 증폭 (Power Pooling)
    # 부호 유지하며 p제곱 수행
    signed_power = np.sign(arr) * (np.abs(arr) ** p)
    
    # 3. 합산 및 정규화
    integrated_vec = np.sum(signed_power, axis=0)
    
    # 코사인 유사도 검색을 위한 L2 정규화
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
        # chunks: List[str] = []
    embedding_list = [] # 임베딩된 벡터값들만 담을 리스트
    current_chunk = ""
    
    for text in texts:
        if len(current_chunk) + len(text) > TOKEN_LIMIT:
            if current_chunk.strip():
                # [즉시 임베딩] 지금까지 모인 청크를 임베딩하여 벡터 저장
                vector = await client.feature_extraction(current_chunk.strip())
                embedding_list.append(vector)
                current_chunk = ""

            # 1-2. [중요] 새로 들어온 text 자체가 limit보다 크다면? 
            # 이 text를 limit 단위로 쪼개서 즉시 임베딩 리스트에 넣음
            if len(text) > TOKEN_LIMIT:
                sub_chunks = [text[i : i + TOKEN_LIMIT] for i in range(0, len(text), TOKEN_LIMIT)]
                # 마지막 조각은 다음 text와 합치기 위해 남겨두고 나머지는 즉시 임베딩
                for sub in sub_chunks[:-1]:
                    vector = await client.feature_extraction(sub.strip())
                    embedding_list.append(vector)
                current_chunk = sub_chunks[-1] # 마지막 조각만 유지
            else:
                current_chunk = text

        else:
            current_chunk += " " + text
    
    if current_chunk.strip():
        vector = await client.feature_extraction(current_chunk.strip())
        embedding_list.append(vector)
        # chunks.append(current_chunk.strip())

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
    
    # 3. 파일 다운로드
    response = requests.get(download_url)
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
@router.post("/embed-batch")
async def get_batch_embeddings(request: dict):
    paths = request.get("paths", [])
    chapter_vectors = []

    # 자바가 보낸 경로 리스트를 순회하며 임베딩
    for path in paths:
        chapter_vec = await core_embedding_logic(path)
        if chapter_vec is not None:
            chapter_vectors.append(chapter_vec)

    if not chapter_vectors:
        raise HTTPException(status_code=400, detail="임베딩할 수 있는 데이터가 없습니다.")

    # 북 벡터 계산 (모든 챕터 벡터의 평균)
    # average_vector = np.mean(chapter_vectors, axis=0)
    average_vector = aggregate_vectors(chapter_vectors, 1)

    book_vector = average_vector.tolist() if hasattr(average_vector, "tolist") else average_vector

    return {
        "book_vector": book_vector,
        "chapter_vectors": [cv.tolist() if hasattr(cv, "tolist") else cv for cv in chapter_vectors]
    }