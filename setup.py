from setuptools import setup, find_packages

setup(
    name="qwen_tts_wrapper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "python-dotenv",
        "numpy",
        "soundfile",
    ],
    author="uwseoul",
    description="Qwen3-TTS-12Hz wrapper for easy integration",
    python_requires=">=3.8",
)
