"""The only module that talks to the network."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from typesafe_sdk import (
    Noul,
    RetryPolicy,
    Score,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
)

Question = Score | Noul


class SystemOneCaller(Protocol):
    def system_one(self, state: dict, questions: Mapping[str, Question], **kwargs) -> SystemOneResponse: ...


class MissingAnswerError(RuntimeError):
    """The response omitted a question we asked. That is a bug, not a condition."""


@dataclass(frozen=True)
class Judgment:
    model: str
    input_tokens: int
    request_id: str | None
    answers: dict[str, dict]
    cached: bool


def _questions_json(questions: Mapping[str, Question]) -> dict:
    return {qid: q.model_dump(mode="json") for qid, q in sorted(questions.items())}


def cache_key(state: dict, questions: Mapping[str, Question], model: str | None) -> str:
    blob = json.dumps(
        {"state": state, "questions": _questions_json(questions), "model": model},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class Judge:
    def __init__(
        self,
        client: SystemOneCaller | None = None,
        cache_path: str | Path | None = ".jev-cache.json",
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else TypeSafeClient(model=model, retry=retry or RetryPolicy(max_retries=3))
        )
        self._cache_path = Path(cache_path) if cache_path else None
        self._model = model

    def judge(self, state: dict, questions: Mapping[str, Question]) -> Judgment:
        key = cache_key(state, questions, self._model)
        cache = self._read_cache()
        if key in cache:
            stored = cache[key]
            return Judgment(
                model=stored["model"],
                input_tokens=stored["input_tokens"],
                request_id=stored["request_id"],
                answers=stored["answers"],
                cached=True,
            )

        response = self._client.system_one(state, questions)
        answers = {qid: answer.model_dump(mode="json") for qid, answer in response.answers.items()}
        missing = sorted(set(questions) - set(answers))
        if missing:
            raise MissingAnswerError(f"response omitted answers for: {', '.join(missing)}")

        try:
            request_id = response.request_id
        except TypeSafeError:
            request_id = None

        judgment = Judgment(
            model=response.model,
            input_tokens=response.usage.input_tokens or 0,
            request_id=request_id,
            answers=answers,
            cached=False,
        )
        cache[key] = {
            "model": judgment.model,
            "input_tokens": judgment.input_tokens,
            "request_id": judgment.request_id,
            "answers": judgment.answers,
        }
        self._write_cache(cache)
        return judgment

    def _read_cache(self) -> dict:
        if not self._cache_path or not self._cache_path.exists():
            return {}
        return json.loads(self._cache_path.read_text())

    def _write_cache(self, cache: dict) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
