import base64
import math
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class InputAudioPayload(BaseModel):
    data: str
    format: str


class ChatCompletionContentPartInputAudioParam(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudioPayload


class ChatCompletionUserMessageParam(BaseModel):
    role: Literal["user"]
    content: list[ChatCompletionContentPartInputAudioParam]


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatCompletionUserMessageParam]
    stream: bool = False
    language: str = "None"
    batch_size: int = Field(default=32, gt=0)
    timestamp: Literal["chunk", "word"] = "chunk"


class ChatCompletionAssistantMessageParam(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionAssistantMessageParam
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionsResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: "ChatCompletionUsage"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: None = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "openai"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


def _decode_audio_data(encoded_audio: str) -> bytes:
    raw_data = encoded_audio
    if "," in encoded_audio and ";base64" in encoded_audio.split(",", 1)[0]:
        raw_data = encoded_audio.split(",", 1)[1]

    try:
        return base64.b64decode(raw_data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc


def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        text_value = output.get("text")
        if isinstance(text_value, str):
            return text_value
        return str(output)
    if isinstance(output, str):
        return output
    return str(output)


def _usage_from_output(output: Any, extracted_text: str) -> ChatCompletionUsage:
    prompt_seconds = 1.0
    if isinstance(output, dict):
        chunks = output.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                timestamp = chunk.get("timestamp")
                if not isinstance(timestamp, (list, tuple)):
                    continue

                prompt_seconds = max(prompt_seconds, *[as_float(ts) for ts in timestamp])

    prompt_tokens = int(math.ceil(prompt_seconds))
    completion_tokens = int(math.ceil(len(extracted_text) / 4))
    return ChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def as_float(n: any, default: float = 0.0) -> float:
    if n is None or not isinstance(n, (int, float)):
        return default
    try:
        return float(n)
    except Exception:
        return default


def create_openai_router(
    transcribe_fn: Callable[..., Any],
    whisper_model_id: str,
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models", response_model=ModelsResponse)
    def list_models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=whisper_model_id,
                )
            ]
        )

    @router.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
    def chat_completions(payload: ChatCompletionsRequest) -> ChatCompletionsResponse:
        if payload.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")

        audio_part: ChatCompletionContentPartInputAudioParam | None = None
        for message in payload.messages:
            for part in message.content:
                if part.type == "input_audio":
                    audio_part = part
                    break
            if audio_part is not None:
                break

        if audio_part is None:
            raise HTTPException(status_code=400, detail="No input_audio content part provided")

        audio_bytes = _decode_audio_data(audio_part.input_audio.data)
        extension = audio_part.input_audio.format.strip().lower() or "wav"

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name

            output = transcribe_fn(
                url=temp_path,
                task="transcribe",
                language=payload.language,
                batch_size=payload.batch_size,
                timestamp=payload.timestamp,
                diarise_audio=False,
                webhook=None,
                task_id=None,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

        extracted_text = _extract_text(output)
        return ChatCompletionsResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=payload.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionAssistantMessageParam(
                        content=extracted_text,
                    ),
                )
            ],
            usage=_usage_from_output(output, extracted_text),
        )

    return router
