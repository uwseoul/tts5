import os
import shutil
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from preprocess import preprocessor
import run_pipeline

class TTSManager:
    """
    다른 프로젝트에서 TTS 기능을 쉽게 사용할 수 있도록 돕는 통합 관리 클래스
    """
    def __init__(
        self, 
        model_size: str = "0.6b", 
        use_mlx: bool = True,
        default_speaker: str = "Serena"
    ):
        self.model_size = model_size
        self.use_mlx = use_mlx and run_pipeline.MLX_AVAILABLE
        self.default_speaker = default_speaker
        self.output_base = "outputs"
        
        if not self.use_mlx:
            print("⚠️ MLX unavailable or disabled. Falling back to PyTorch.")

    def generate(
        self, 
        text: str, 
        speaker: Optional[str] = None, 
        ref_audio: Optional[str] = None,
        voice_design: Optional[str] = None,
        output_name: str = "final.wav",
        max_chars: int = 500
    ) -> str:
        """
        텍스트를 입력받아 최종 음성 파일 경로를 반환합니다.
        """
        # 1. 출력 디렉토리 생성
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        job_dir = os.path.join(self.output_base, f"job_{timestamp}")
        os.makedirs(job_dir, exist_ok=True)

        # 2. 텍스트 전처리
        print(f"Preprocessing text ({len(text)} chars)...")
        doc = preprocessor.preprocess_to_json(text, max_chars=max_chars)
        segments = doc.get("segments", [])

        # 3. 음성 생성
        print(f"Generating voice using {self.model_size} model...")
        audio_files, _ = run_pipeline.generate_voice(
            segments=segments,
            output_dir=job_dir,
            speaker=speaker or self.default_speaker,
            ref_audio=ref_audio,
            model_size=self.model_size,
            use_mlx=self.use_mlx,
            voice_design=voice_design
        )

        if not audio_files:
            raise RuntimeError("Audio generation failed.")

        # 4. 오디오 결합
        final_path = run_pipeline.merge_audio(audio_files, job_dir)
        
        if not final_path:
            raise RuntimeError("Audio merging failed.")

        # 5. 지정된 이름으로 이동 (선택 사항)
        destination = os.path.join(job_dir, output_name)
        if final_path != destination:
            shutil.move(final_path, destination)
            
        print(f"✅ TTS Generation Complete: {destination}")
        return destination

if __name__ == "__main__":
    # 간단한 테스트
    manager = TTSManager()
    # manager.generate("안녕하세요, 통합 모듈 테스트입니다.", output_name="test_result.wav")
