\# 개발 환경 설치



현재 검증 환경:

\- Windows

\- Python 3.14.6

\- NVIDIA RTX 5080

\- NVIDIA 드라이버 610.88

\- PyTorch 2.14.0+cu130



가상환경을 만든 뒤 GPU용 PyTorch를 먼저 설치한다.

python -m pip install "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130

python -m pip install -r requirements.txt

