# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio profiling must respect the processor's configured chunk length."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import AudioProcessorItems

from ...utils import build_model_context

_QWEN2_AUDIO = "Qwen/Qwen2-Audio-7B-Instruct"
_AUDIO_ONLY_LIMITS = {"audio": 1, "image": 0, "video": 0}


@pytest.mark.parametrize(("chunk_length", "expected_tokens"), [(30, 750), (31, 775)])
def test_qwen2_audio_token_budget(chunk_length: int, expected_tokens: int) -> None:
    ctx = build_model_context(
        _QWEN2_AUDIO,
        mm_processor_kwargs={"chunk_length": chunk_length},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    assert processor.info.get_mm_max_tokens_per_item(8192, {"audio": 1}) == {
        "audio": expected_tokens,
    }


@pytest.mark.parametrize(
    ("model_id", "chunk_length", "mm_processor_kwargs"),
    [
        (_QWEN2_AUDIO, 31, {"chunk_length": 31}),
        ("Qwen/Qwen2.5-Omni-3B", 300, None),
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", 30, None),
        ("moonshotai/Kimi-Audio-7B-Instruct", 31, {"chunk_length": 31}),
        ("Qwen/Qwen3-ASR-0.6B", 31, {"chunk_length": 31}),
    ],
)
def test_audio_dummy_uses_processor_chunk_length(
    model_id: str,
    chunk_length: int,
    mm_processor_kwargs: dict[str, int] | None,
) -> None:
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=_AUDIO_ONLY_LIMITS,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    dummy = processor.get_dummy_inputs(8192, {"audio": 1}, {})

    audio = dummy.mm_data_items.get_items("audio", AudioProcessorItems)
    assert audio.get_audio_length(0) == chunk_length * 16000
