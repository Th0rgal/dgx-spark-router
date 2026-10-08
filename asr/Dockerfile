FROM vllm-skinny-tp1:v1
# Audio decode deps for /v1/audio/transcriptions (vllm[audio]); numpy pinned to the base version.
RUN pip install --no-cache-dir "numpy==2.2.6" av scipy soundfile soxr librosa "mistral_common[audio]" \
 && python3 -c "import soundfile, soxr, av, librosa; print('audio deps ok')"
