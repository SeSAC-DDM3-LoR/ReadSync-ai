# 1. 베이스 이미지 설정
FROM python:3.13-slim

# 2. Poetry 설치 및 환경 설정
# virtualenvs.create false: 컨테이너 자체가 격리 환경이므로 가상환경을 따로 만들지 않습니다.
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/app/src

# 3. 필수 도구 설치
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 의존성 파일 먼저 복사 (캐싱 활용)
COPY pyproject.toml poetry.lock* ./

# 6. 패키지 설치
# --no-root: 현재 프로젝트 자체를 패키지로 설치하지 않음
# --without dev: 아까 설치한 types-aioboto3 같은 개발용 패키지 제외
RUN poetry install --no-root --without dev

# 7. 전체 소스 코드 복사
COPY . .

# 8. 포트 개방 (8000번 사용하신다고 하셨으니)
EXPOSE 8000

# 9. 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]