"""Real instruct-LLM lyric generation for the ``generated`` source (US4, FR-011, Decision L1).

Runs an open instruct LLM (via ``transformers``) to produce themed lyric text for a score; the
``generated`` source then segments it into one syllable per note (FR-019) and pins the result
(FR-012), so the LLM's non-determinism is absorbed by the cache rather than fought. Heavy imports
(``torch``/``transformers``) are lazy, so the CPU baseline and non-generated runs never load them
(FR-005). Needs the ``gpu`` + ``lyrics-gpu`` extras.
"""

from __future__ import annotations

import functools

# A small, permissively licensed instruct model is plenty: the goal is singable themed syllables for
# phonetic diversity, not literary quality. Override per run via ``LyricModel.model_ref``.
DEFAULT_MODEL_REF = "Qwen/Qwen2.5-0.5B-Instruct"


@functools.cache
def _load(model_ref: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=dtype).to(device)
    model.eval()
    return tokenizer, model, device


def generate_lyrics(
    theme: str,
    n_syllables: int,
    seed: int,
    *,
    model_ref: str = "",
    max_new_tokens: int = 160,
) -> str:
    """Generate themed lyric text aiming for about ``n_syllables`` singable syllables (FR-011).

    Returns raw text; the caller segments it into one-syllable-per-note and pins it (FR-012/FR-019).
    """
    import torch

    ref = model_ref or DEFAULT_MODEL_REF
    tokenizer, model, device = _load(ref)
    torch.manual_seed(seed)

    target = max(1, n_syllables)
    theme_text = theme.strip() or "a wordless melody"
    prompt = (
        f"Write simple, singable English lyrics about {theme_text}. "
        f"Use about {target} short one- or two-syllable words. "
        "Reply with only the words on one line, separated by spaces, with no punctuation, "
        "no title, and no explanation."
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()
