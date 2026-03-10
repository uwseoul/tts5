# TTS 프로그램 기능 작동 순서

## 📋 전체 워크플로우

> [!IMPORTANT]
> **프로젝트 지침**: 속도 및 최적화 이슈로 인하여 PyTorch 백엔드 생성을 금지합니다. 모든 음성 생성(TTS, STS, Voice Cloning 등)은 반드시 Apple Silicon 전용 **`mlx-audio`** 패키지를 기반으로 동작해야 합니다.

```
원본 텍스트 → 로컬 정제 → LLM 전처리 (GLM-4.5-Air) → 감정/지시어 추출 → 음성 생성 (Qwen3-TTS via MLX) → 청크 결합 (FFmpeg) → 리포트 생성
```

---

## 🔄 단계별 상세 설명

### **STEP 1: 환경 설정 및 준비**

#### 1-1. API 키 설정 (`.env`)
- **파일**: `.env` (프로젝트 루트)
- **내용**: 코딩플랜 API 키 및 엔드포인트 설정
  ```env
  OPENAI_API_KEY=your_api_key
  OPENAI_API_BASE=https://api.z.ai/api/coding/paas/v4
  LLM_MODEL=glm-4.5-air
  ```

#### 1-2. 입력 파일 준비
- **위치**: `sample/sample.txt` 등 `.txt` 파일
- **역할**: TTS로 변환할 원본 스크립트

---

### **STEP 2: 고도화된 텍스트 전처리 (LLM)**

#### 2-1. 실행 방식
```bash
python run_pipeline.py sample/sample.txt
```

#### 2-2. 전처리 프로세스
1. **로컬 마크다운 정제**: 헤딩, 볼드, 이모지, 시간대 메타데이터 등 제거
2. **LLM 청크 분할 (GLM-4.5-Air)**:
   - 큰 텍스트를 약 400~800자 단위로 나누어 LLM에 전달
   - **의미적 완결성**과 **낭독 호흡**을 기준으로 세그먼트 분할
3. **감정 및 지시어 생성**:
   - 각 세그먼트마다 문맥에 맞는 `emotion`(활기참, 진지함 등)과 `instruction`(여성, 30대, 신뢰감 있는 톤 등) 부여

#### 2-3. 결과물
- **파일**: `outputs/YYYY-MM-DD_HHMMSS/preprocessed.json`
- **구조**: `[{ "id": 1, "text": "...", "emotion": "...", "instruction": "..." }, ...]`

---

### **STEP 3: 지시어 기반 음성 생성 (Qwen3-TTS)**

#### 3-1. 지시어 매핑 (Instruction Mapping)
전처리에서 생성된 데이터를 아래와 같이 결합하여 모델에 전달합니다:
- **최종 지시어**: `"{emotion} 분위기로, {instruction}"`
- 예: `"진지함 분위기로, 중년 남성, 차분하고 논리적인 톤"`

#### 3-2. 음성 합성 모델 (MLX 전용)
- **모델**: `mlx-community/Qwen3-TTS` 양자화 모델 (0.6B-4bit 또는 1.7B-8bit 등)
- **방식**: `mlx_audio`의 `generate_audio` 함수 사용
- **특징**: PyTorch 대비 10배 이상 압도적인 속도로 Apple Silicon(MPS)을 활용하여 생성

---

### **STEP 4: 청크 결합 및 리포트**

#### 4-1. 결합 (FFmpeg)
- 세그먼트별로 생성된 `chunk_n.wav` 파일들을 하나로 연결하여 `final.wav` 생성

#### 4-2. 결과 리포트
- **파일**: `outputs/YYYY-MM-DD_HHMMSS/report.md`
- 전체 실행 시간, 사용된 모델, 세그먼트별 감정 분석 결과 요약 제공

---

## 🚀 실행 옵션

- **기본 실행**: `python run_pipeline.py 스크립트.txt`
- **스피커 지정**: `python run_pipeline.py 스크립트.txt --speaker Serena`
- **보이스 클로닝**: `python run_pipeline.py 스크립트.txt --ref-audio 샘플.wav`

---

## � 주요 설정 값

- **청크 크기**: LLM 호출 시 400~800자 (타임아웃 방지용)
- **출력 세그먼트**: 각 250~300자 내외 (호흡 최적화)
- **타임아웃**: LLM API 호출 당 120초
