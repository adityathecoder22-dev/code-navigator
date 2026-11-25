"""Shared fixtures. Tests run fully offline: mock LLM, hash embeddings, numpy vector store."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code_navigator.config import Settings  # noqa: E402
from code_navigator.llm.providers import LLM  # noqa: E402

SAMPLE = ROOT / "examples" / "sample-shop"


@pytest.fixture(scope="session")
def offline_settings(tmp_path_factory) -> Settings:
    return Settings.load(None).with_overrides(
        cache_dir=tmp_path_factory.mktemp("cache"),
        llm_provider="none", llm_model="none",
        embed_provider="hash", embed_model="hash-512",
        vector_backend="simple", summary_mode="heuristic",
        retrieval_mode="hybrid", top_k=8, include_tests=True,
    )


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory) -> Path:
    """A throw-away copy of the sample system, so tests never write into examples/."""
    dest = tmp_path_factory.mktemp("repo") / "sample-shop"
    shutil.copytree(SAMPLE, dest)
    return dest


@pytest.fixture(scope="session")
def sample_index(sample_repo, offline_settings):
    from code_navigator.indexing.pipeline import build_index
    return build_index(str(sample_repo), settings=offline_settings, progress=lambda *a: None)


@pytest.fixture(scope="session")
def engine(sample_index, sample_repo, offline_settings):
    from code_navigator.engine import Navigator
    return Navigator(sample_repo, settings=offline_settings)


class ScriptedLLM(LLM):
    """Fake LLM that answers navigation prompts by picking options mentioning given keywords."""
    provider = "scripted"

    def __init__(self, keywords: list[str]):
        super().__init__(model="scripted")
        self.keywords = [k.lower() for k in keywords]
        self.prompts: list[str] = []

    def _complete(self, prompt, system, max_tokens, temperature, json_mode):
        self.prompts.append(prompt)
        if not json_mode:
            return "Scripted answer [1]."
        picked = []
        for line in prompt.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0][:1] in "SCF" and parts[0][1:].isdigit():
                col = {"S": 1, "C": 2, "F": 3}[parts[0][0]]
                name_col = parts[col] if col < len(parts) else ""
                if any(k in name_col.lower() for k in self.keywords):
                    picked.append(parts[0])
        return json.dumps({"selected": picked[:3], "scope": "cross_service", "confidence": 0.9,
                           "reason": "scripted"})


@pytest.fixture
def scripted_llm():
    return ScriptedLLM
