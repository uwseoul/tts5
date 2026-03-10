import os
import sys
import glob
import argparse
import json
import subprocess
import importlib
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from preprocess import preprocessor

MAX_CHARS = 500
LANGUAGE = "Korean"
DEFAULT_SPEAKER = "Serena"
DEFAULT_REF_AUDIO = "sample/ref.wav"
DEFAULT_MODEL_SIZE = "0.6b"

def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except ImportError:
        pass

# mlx-audio 가용 여부 감지
try:
    from mlx_audio.tts.utils import load_model as mlx_load_model
    from mlx_audio.tts.generate import generate_audio as mlx_generate_audio
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# 모델 매핑 (PyTorch 원본 / MLX 최적화)
MODELS = {
    "0.6b": {
        "custom": {
            "pytorch": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "mlx":     "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        },
        "base": {
            "pytorch": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "mlx":     "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        },
    },
    "1.7b": {
        "custom": {
            "pytorch": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "mlx":     "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        },
        "base": {
            "pytorch": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "mlx":     "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
        },
        "voice_design": {
            "pytorch": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "mlx":     "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        },
    },
}


def create_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {name}") from e


def _load_pytorch_model(model_id: str):
    mod = _require_module("qwen_tts.inference.qwen3_tts_model")
    Qwen3TTSModel = getattr(mod, "Qwen3TTSModel")
    return Qwen3TTSModel.from_pretrained(model_id)


# ─── MLX 기반 세그먼트 생성 ─────────────────────────────────────
def _generate_segment_mlx(
    model: Any,
    text: str,
    output_file: str,
    seg_num: int,
    total_segs: int,
    speaker: Optional[str] = None,
    instruction: Optional[str] = None,
    voice_design: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> float:
    sf = _require_module("soundfile")

    if voice_design:
        print(f"  [MLX-VD] Segment {seg_num}/{total_segs}: {len(text)}자 (VoiceDesign)")
    elif ref_audio:
        print(f"  [MLX-VC] Segment {seg_num}/{total_segs}: {len(text)}자 (VoiceClone)")
    else:
        print(f"  [MLX] Segment {seg_num}/{total_segs}: {len(text)}자 (Speaker: {speaker})")

    tmp_dir = output_file + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        mlx_generate_audio(
            model=model,
            text=text,
            voice=None if (voice_design or ref_audio) else (speaker or "Sohee"),
            instruct=voice_design or instruction or "",
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=1.0,
            lang_code="korean",
            temperature=0.0,   # greedy → 세그먼트 간 목소리 일관성 확보
            output_path=tmp_dir,
        )

        # 생성된 wav 파일을 output_file 위치로 이동
        wav_paths = glob.glob(os.path.join(tmp_dir, "*.wav"))
        if not wav_paths:
            print(f"  ⚠️ MLX 오디오 파일 생성 안 됨: {tmp_dir} - 해당 세그먼트 무시")
            return 0.0

        import shutil
        shutil.move(wav_paths[0], output_file)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        audio_array, sample_rate = sf.read(output_file)
        return len(audio_array) / sample_rate

    except Exception as e:
        print(f"  ❌ 세그먼트 생성 실패 (예외 발생): {e}")
        return 0.0

# ─── PyTorch 기반 세그먼트 생성 ──────────────────────────────────
def _generate_segment_pytorch(
    model: Any,
    text: str,
    output_file: str,
    seg_num: int,
    total_segs: int,
    speaker: Optional[str] = None,
    voice_clone_prompt: Optional[Any] = None,
    instruction: Optional[str] = None,
) -> float:
    np = _require_module("numpy")
    sf = _require_module("soundfile")

    model_type = getattr(model.model, "tts_model_type", "base")

    if voice_clone_prompt is not None:
        print(f"  [PT] Segment {seg_num}/{total_segs}: {len(text)}자 (Voice Clone)")
        output = model.generate_voice_clone(
            text=text, voice_clone_prompt=voice_clone_prompt,
            language=LANGUAGE, instruction=instruction,
        )
    elif model_type == "voice_design":
        print(f"  [PT] Segment {seg_num}/{total_segs}: {len(text)}자 (VoiceDesign)")
        design_instruct = f"{speaker} 목소리로. {instruction}" if instruction else speaker
        output = model.generate_voice_design(
            text=text, instruct=design_instruct, language=LANGUAGE,
        )
    else:
        print(f"  [PT] Segment {seg_num}/{total_segs}: {len(text)}자 (Speaker: {speaker})")
        output = model.generate_custom_voice(
            text=text, speaker=speaker, language=LANGUAGE, instruction=instruction,
        )

    audio_array = np.array(output[0][0], dtype=np.float32)
    sf.write(output_file, audio_array, output[1])
    return len(audio_array) / output[1]


def generate_voice(
    segments: List[Dict[str, Any]],
    output_dir: str,
    speaker: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    model_size: str = "0.6b",
    limit: Optional[int] = None,
    use_mlx: Optional[bool] = None,
    voice_design: Optional[str] = None,
) -> Tuple[List[str], float]:

    config = MODELS.get(model_size, MODELS["0.6b"])

    # backend_type 결정
    if voice_design:
        backend_type = "voice_design"
        if backend_type not in config:
            config = MODELS["1.7b"]
            model_size = "1.7b"
            print(f"  VoiceDesign 모델은 1.7b만 지원. 자동 전환.")
    elif ref_audio:
        backend_type = "base"
    else:
        backend_type = "custom"

    if use_mlx is None:
        use_mlx = MLX_AVAILABLE
    if use_mlx and not MLX_AVAILABLE:
        print("⚠️ MLX 미설치 - PyTorch로 fallback")
        use_mlx = False

    model_id = config[backend_type]["mlx" if use_mlx else "pytorch"]
    engine = "Apple Silicon (MLX)" if use_mlx else f"{detect_device()} (PyTorch)"
    print(f"장치: {engine}")
    print(f"모델: {model_id} ({model_size})")
    if voice_design:
        print(f"보이스 디자인: {voice_design}")
    elif ref_audio:
        print(f"참조 오디오: {ref_audio}")
    else:
        print(f"스피커: {speaker}")

    print("모델 로드 중...")
    if use_mlx:
        model = mlx_load_model(model_id)
        voice_clone_prompt = None
        if ref_audio and not ref_text:
            print("참조 오디오 대본을 추출합니다 (최초 1회)...")
            from mlx_audio.stt import load as load_stt_model
            from mlx_audio.utils import load_audio
            import mlx.core as mx
            stt_model = load_stt_model("mlx-community/whisper-large-v3-turbo-asr-fp16")
            audio_stt = load_audio(ref_audio, sample_rate=16000)
            ref_text = stt_model.generate(audio_stt).text
            del stt_model
            mx.clear_cache()
            print(f"✅ 추출된 대본: {ref_text}")
    else:
        model = _load_pytorch_model(model_id)
        voice_clone_prompt = None
        if ref_audio:
            print("참조 오디오 특징 추출 중...")
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio, x_vector_only_mode=True
            )
            print("✅ 특징 추출 완료")

    print("✅ 모델 로드 완료")

    if limit:
        segments = segments[:limit]
        print(f"  (테스트 모드: 상위 {limit}개 세그먼트만 생성)")

    total = len(segments)
    if total == 0:
        return [], 0.0

    audio_files: List[str] = []
    total_duration = 0.0

    for i, seg in enumerate(segments, 1):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        seg_speaker = seg.get("speaker", speaker)
        seg_emotion = seg.get("emotion", "")
        seg_instruction = seg.get("instruction", "")

        if seg_emotion and seg_instruction:
            full_instruction = f"{seg_emotion} 분위기로, {seg_instruction}"
        elif seg_instruction:
            full_instruction = seg_instruction
        elif seg_emotion:
            full_instruction = seg_emotion
        else:
            full_instruction = ""

        out = os.path.join(output_dir, f"chunk_{i}.wav")

        if use_mlx:
            duration = _generate_segment_mlx(
                model=model, text=text, output_file=out,
                seg_num=i, total_segs=total,
                speaker=seg_speaker,
                instruction=full_instruction or None,
                voice_design=voice_design,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            if duration > 0:
                total_duration += duration
                audio_files.append(out)
        else:
            duration = _generate_segment_pytorch(
                model=model, text=text, output_file=out,
                seg_num=i, total_segs=total,
                speaker=seg_speaker,
                voice_clone_prompt=voice_clone_prompt,
                instruction=full_instruction or None,
            )
            if duration > 0:
                total_duration += duration
                audio_files.append(out)

    return audio_files, total_duration


def merge_audio(audio_files: List[str], output_dir: str) -> Optional[str]:
    if not audio_files:
        return None

    concat_file = os.path.join(output_dir, "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for fpath in audio_files:
            f.write(f"file '{os.path.abspath(fpath)}'\n")

    final_wav = os.path.join(output_dir, "final.wav")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", final_wav]

    print("\nWAV 결합 중...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ WAV 결합 완료: {final_wav}")
        return final_wav
    else:
        print(f"❌ WAV 결합 실패:\n{result.stderr}")
        return None


def main(
    input_file: str,
    max_chars: Optional[int] = None,
    speaker: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    model_size: str = "0.6b",
    limit: Optional[int] = None,
    use_mlx: Optional[bool] = None,
    voice_design: Optional[str] = None,
    seed: int = 42,
) -> None:
    set_seed(seed)
    active_ref_audio = ref_audio
    active_speaker = speaker if (not active_ref_audio and not voice_design) else None
    if not active_ref_audio and not active_speaker and not voice_design:
        active_speaker = DEFAULT_SPEAKER

    start_time = datetime.now()
    backend_label = "MLX" if (use_mlx or (use_mlx is None and MLX_AVAILABLE)) else "PyTorch"
    print("=" * 60)
    if voice_design:
        mode_str = f"VoiceDesign: {voice_design[:40]}"
    elif active_ref_audio:
        mode_str = f"Voice Clone: {active_ref_audio}"
    else:
        mode_str = f"CustomVoice: {active_speaker}"
    print(f"TTS 파이프라인 시작 ({model_size} | {mode_str} | {backend_label})")
    print("=" * 60)

    output_dir = create_output_dir()
    print(f"\n[1/5] 출력 디렉토리 생성...\n출력 디렉토리: {output_dir}")

    chunk_limit = max_chars if max_chars else MAX_CHARS
    doc = None
    preprocessed_json_path = input_file

    if input_file.endswith(".txt"):
        print(f"\n[2/5] 텍스트 전처리 중... ({input_file})")
        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        doc = preprocessor.preprocess_to_json(raw_text, chunk_limit)
        preprocessed_json_path = os.path.join(output_dir, "preprocessed.json")
        with open(preprocessed_json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    elif input_file.endswith(".json"):
        print("\n[2/5] 전처리 JSON 로드 중...")
        with open(input_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        print("❌ 지원하지 않는 파일 형식입니다.")
        sys.exit(1)

    segments = doc.get("segments", [])
    total_chars = sum(len(seg.get("text", "")) for seg in segments)
    print(f"전처리 후 글자 수: {total_chars}자")
    print(f"세그먼트 개수: {len(segments)}개")

    print("\n[3/5] 음성 생성 중...")
    gen_start = datetime.now()
    audio_files, total_duration = generate_voice(
        segments, output_dir,
        speaker=active_speaker, ref_audio=active_ref_audio, ref_text=ref_text,
        model_size=model_size, limit=limit, use_mlx=use_mlx,
        voice_design=voice_design,
    )
    gen_end = datetime.now()
    gen_time = (gen_end - gen_start).total_seconds()

    if not audio_files:
        print("\n❌ 음성 생성 실패")
        sys.exit(1)

    print("\n[4/5] 청크 결합 중...")
    final_wav = merge_audio(audio_files, output_dir)

    print("\n[5/5] 리포트 작성 중...")
    report_file = os.path.join(output_dir, "report.md")
    timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# TTS 생성 리포트\n\n")
        f.write(f"**생성 시각**: {timestamp}\n\n")
        f.write("## 설정\n\n")
        config = MODELS.get(model_size, MODELS["0.6b"])
        if voice_design:
            backend_type = "voice_design"
            if backend_type not in config:
                config = MODELS["1.7b"]
        elif active_ref_audio:
            backend_type = "base"
        else:
            backend_type = "custom"
        b = "mlx" if (use_mlx or (use_mlx is None and MLX_AVAILABLE)) else "pytorch"
        model_id = config[backend_type][b]
        f.write(f"- **백엔드**: {b.upper()}\n")
        f.write(f"- **모델**: {model_id}\n")
        f.write(f"- **언어**: {LANGUAGE}\n")
        if voice_design:
            f.write(f"- **보이스 디자인**: {voice_design}\n")
        elif active_ref_audio:
            f.write(f"- **참조 오디오**: {active_ref_audio}\n")
        else:
            f.write(f"- **스피커**: {active_speaker}\n")
        f.write(f"- **최대 세그먼트 길이**: {chunk_limit}자\n\n")

        f.write("## 메트릭\n\n")
        f.write(f"- **전처리 후 글자 수**: {total_chars}자\n")
        f.write(f"- **세그먼트 개수**: {len(segments)}개\n\n")

        f.write("## 세그먼트별 정보\n\n")
        f.write("| 세그먼트 | 글자 수 | 감정 | 음성 파일 |\n")
        f.write("|----------|--------|------|----------|\n")
        for i, seg in enumerate(segments, 1):
            text = seg.get("text", "")
            emotion = seg.get("emotion", "N/A")
            f.write(f"| {i} | {len(text)} | {emotion} | chunk_{i}.wav |\n")

        f.write("\n## 최종 결과\n\n")
        f.write(f"- **최종 파일**: {os.path.basename(final_wav) if final_wav else 'N/A'}\n")
        f.write(f"- **총 음성 길이**: {total_duration:.2f}초\n")
        if total_duration > 0:
            f.write(f"- **평균 낭독 속도**: {total_chars / total_duration:.1f}자/초\n")
        if gen_time > 0:
            gen_efficiency = total_duration / gen_time
            f.write(f"- **생성 효율**: {gen_efficiency:.2f}배속\n")
            f.write(f"- **순수 생성 소요 시간**: {gen_time:.2f}초\n")

        total_time = (datetime.now() - start_time).total_seconds()
        mins, secs = divmod(int(total_time), 60)
        f.write(f"- **총 소요 시간**: {mins}분 {secs}초\n")

    print(f"✅ 리포트 완료: {report_file}")
    print("\n" + "=" * 60)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 60)
    print(f"출력 디렉토리: {output_dir}")
    if final_wav:
        print(f"최종 파일: {final_wav}")
    print(f"리포트: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS 파이프라인")
    parser.add_argument("input_file", help="입력 텍스트(.txt) 또는 전처리 JSON(.json)")
    parser.add_argument("--max-chars", type=int, help=f"최대 세그먼트 길이 (기본 {MAX_CHARS})")
    parser.add_argument("--speaker", type=str, help=f"스피커 이름 (기본: {DEFAULT_SPEAKER})")
    parser.add_argument("--ref-audio", type=str, help="음성 복제용 참조 오디오")
    parser.add_argument("--ref-text", type=str, help="음성 복제용 참조 오디오의 대본 (없으면 자동 변환)")
    parser.add_argument("--model-size", type=str, choices=["0.6b", "1.7b"], default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--limit", type=int, help="생성할 세그먼트 개수 제한 (테스트용)")
    parser.add_argument("--no-mlx", action="store_true", help="MLX 비활성화, PyTorch 강제 사용 (비권장)")
    parser.add_argument("--voice-design", type=str, help="보이스 디자인 프롬프트 (예: '한국 30대 여자 뉴스 아나운서')")
    parser.add_argument("--seed", type=int, default=42, help="난수 고정 시드 (기본: 42)")
    args = parser.parse_args()

    # 프로젝트 지침: PyTorch보다 MLX 기반 오디오 생성을 기본(우선)으로 사용
    use_mlx = False if args.no_mlx else True
    main(args.input_file, args.max_chars, args.speaker, args.ref_audio, args.ref_text, args.model_size, args.limit, use_mlx, args.voice_design, args.seed)
