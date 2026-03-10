import os
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def clean_markdown(text: str) -> str:
    """마크다운 특수문자 및 불필요한 공백 제거"""
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    return text.strip()

def split_text_by_llm(text: str, max_chars: int = 500) -> Optional[List[Dict[str, Any]]]:
    """GLM-4.5-Air API를 호출하여 텍스트를 최적의 세그먼트로 분할"""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.z.ai/api/coding/paas/v4")
    model = os.getenv("LLM_MODEL", "glm-4.5-air")

    if not api_key:
        print("⚠️ OPENAI_API_KEY 환경 변수가 없습니다. LLM 대신 로컬 정규식 분할을 사용합니다.")
        return None

    # 시스템 프롬프트 확정 (지시어 및 감정 제거 - 2~4문장 단위 낭독용)
    system_prompt = """
당신은 TTS 낭독용 텍스트 전처리기입니다.
목표는 원문을 자연스럽게 읽히는 낭독 세그먼트로 정제하는 것입니다.

[우선순위]
1. 본문 외 요소 삭제
2. 리스트/나열을 자연스러운 문장으로 변환
3. 문맥이 이어지도록 세그먼트 병합
4. JSON 형식 준수

[입력 해석 규칙]
- 줄바꿈은 문장 경계가 아닙니다. 문장 경계는 마침표(.), 물음표(?), 느낌표(!)를 기준으로 판단하십시오.
- 제목, 소제목, 장/절 번호, 섹션 번호, 타임스탬프, 구간 시간, 마크다운 기호(#, ##, **, >, *, -, ---)는 모두 삭제하십시오.
- 예: "## 6. 버블 가능성은 없을까? (8:30~9:30)" 같은 섹션 제목은 통째로 삭제하십시오.
- 예: "(0:00~1:00)", "8:30~9:30" 같은 시간 표시는 어디에 있든 삭제하십시오.
- 인용 블록 기호(>)는 삭제하되, 인용문 내용이 본문이면 내용은 유지하십시오.
- 불릿 목록과 숫자 목록은 표식만 지우지 말고, 항목 내용을 살려 낭독 가능한 완전한 문장으로 바꾸십시오.
- 의미를 바꾸지 않는 범위에서 조사, 연결어, 서술어(예: ~입니다, ~이 있습니다, ~가 필요합니다)는 보완할 수 있습니다.

[세그먼트 규칙]
- 기본 단위는 2~4문장입니다.
- 목표 길이는 120~180자입니다.
- 허용 범위는 90~220자입니다.
- 90자 미만 세그먼트는 반드시 앞이나 뒤 세그먼트와 병합하십시오.
- 220자 초과이거나 주제가 명확히 전환될 때만 분할하십시오.
- 질문문, 짧은 강조문, 짧은 한 줄 문장은 단독 세그먼트로 두지 마십시오.
- 마지막 세그먼트만 예외적으로 1문장 또는 60자 이상 허용됩니다.

[나열 처리 예시]
입력: "1. 반도체 (연산 능력) 2. 전력 (에너지) 3. 데이터센터 (물리 인프라)"
출력: "이 계산기를 돌리려면 반도체, 전력, 데이터센터가 필요합니다."

입력: "* GPU 기업 * 전력 설비 기업 * 냉각 시스템 기업 * 고대역폭 메모리 기업"
출력: "예를 들어 GPU 기업, 전력 설비 기업, 냉각 시스템 기업, 고대역폭 메모리 기업이 있습니다."

입력: "특히 영향을 받을 가능성이 높은 분야는: 반복적 사무직 기본 코딩 단순 콘텐츠 제작"
출력: "특히 영향을 받을 가능성이 높은 분야는 반복적 사무직, 기본 코딩, 단순 콘텐츠 제작입니다."

[금지]
- 섹션 제목, 번호, 타임스탬프, 마크다운 기호를 출력에 남기지 마십시오.
- 리스트를 키워드 나열 형태로 남기지 마십시오.
- 새 정보, 요약, 감정, 지시문을 추가하지 마십시오.
- JSON 외의 설명 문장을 출력하지 마십시오.

[출력 형식]
- 반드시 순수 JSON 배열만 출력하십시오.
- 각 원소는 아래 형식만 허용됩니다.
  {"text":"...", "emotion":"", "instruction":""}

[최종 검수]
출력 전 내부적으로 반드시 확인하십시오.
1. 모든 세그먼트가 2~4문장인가?
2. 90자 미만 세그먼트가 남아 있지 않은가?
3. 타임스탬프, 섹션 제목, 번호, 마크다운 기호가 남아 있지 않은가?
4. 리스트가 낭독 가능한 완전한 문장으로 바뀌었는가?
"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 텍스트를 전처리해줘:\n\n{text}"}
        ],
        "temperature": 0.1,
        "top_p": 0.7
    }

    try:
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # 마크다운 블록 제거 및 JSON 추출
        if "```" in content:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1).strip()
            else:
                content = re.sub(r'^```(?:json)?\n?|\n?```$', '', content, flags=re.MULTILINE).strip()
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 보정 시도
            content_fixed = content.strip()
            if not content_fixed.endswith("]"): content_fixed += "]"
            try:
                return json.loads(content_fixed)
            except:
                print(f"❌ JSON 파싱 실패")
                return None
    except Exception as e:
        print(f"❌ LLM API 호출 오류: {e}")
        return None

def preprocess_to_json(text: str, max_chars: int = 500, use_llm: bool = True, source_file: str = "raw_content", output_dir: Optional[str] = None) -> Dict[str, Any]:
    """텍스트 전처리 전체 시퀀스를 수행하여 JSON 객체 반환"""
    # 1. 마크다운 정리
    cleaned_main = clean_markdown(text)
    
    segments = []
    method = "regex_fallback"

    # 2. LLM 전처리 시도
    if use_llm:
        # 대형 청크 분할 (1500자 단위)
        text_chunks = []
        paragraphs = cleaned_main.split('\n\n')
        curr_chunk = ""
        for p in paragraphs:
            if len(curr_chunk) + len(p) < 1500:
                curr_chunk += p + "\n\n"
            else:
                if curr_chunk: text_chunks.append(curr_chunk.strip())
                curr_chunk = p + "\n\n"
        if curr_chunk: text_chunks.append(curr_chunk.strip())

        print(f"  LLM 전처리 시작... (총 {len(text_chunks)}개 청크 병렬 처리 중)")
        
        all_llm_chunks = [None] * len(text_chunks)
        success_all = True
        
        def process_chunk(idx, chunk_text):
            print(f"    청크 {idx+1}/{len(text_chunks)} 처리 중...")
            res = split_text_by_llm(chunk_text, max_chars)
            return idx, res

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(text_chunks)]
            for future in futures:
                idx, res = future.result()
                if res is not None:
                    all_llm_chunks[idx] = res
                else:
                    success_all = False

        if success_all and all(c is not None for c in all_llm_chunks):
            method = "llm_optimized"
            merged_res = []
            for c in all_llm_chunks:
                merged_res.extend(c)
                
            for i, chunk in enumerate(merged_res, 1):
                segments.append({
                    "id": i,
                    "text": chunk.get("text", "").strip(),
                    "emotion": "",
                    "instruction": ""
                })
            
            # 즉시 저장
            if output_dir:
                doc_temp = {
                    "metadata": {
                        "source_file": source_file,
                        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
                        "total_segments": len(segments),
                        "method": method + "_save"
                    },
                    "segments": segments
                }
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "preprocessed.json"), "w", encoding="utf-8") as f:
                    json.dump(doc_temp, f, ensure_ascii=False, indent=2)
                print(f"  ✅ 전처리 결과 즉시 저장 완료: {output_dir}/preprocessed.json")

    # 3. 폴백 (정규식)
    if not segments:
        print("  ⚠️ LLM 실패로 로컬 분할 사용")
        method = "regex_fallback"
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_main)
        for i, s in enumerate(sentences, 1):
            if s.strip():
                segments.append({
                    "id": i,
                    "text": s.strip(),
                    "emotion": "평온함",
                    "instruction": "중립적인 톤"
                })

    return {
        "metadata": {
            "source_file": source_file,
            "preprocessed_at": datetime.now(timezone.utc).isoformat(),
            "total_segments": len(segments),
            "method": method
        },
        "segments": segments
    }

def preprocess_text(text: str, max_chars: int = 500) -> List[Dict[str, Any]]:
    """하위 호환성 유지용"""
    res = preprocess_to_json(text, max_chars, use_llm=False)
    return res["segments"]
