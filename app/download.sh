#!/bin/bash

cd /app/whisper-api
/app/whisper-api/.venv/bin/python -m app.download_weights
/app/whisper-api/.venv/bin/python -m app.download_llm
