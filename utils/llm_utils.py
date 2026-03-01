"""LLM-specific helpers: response delimiters for stripping generated text."""
from typing import Tuple


def get_response_delimiters(llm_id: str) -> Tuple[str, str]:
    """Return (pre_header, post_header) for stripping model response from decoded text."""
    if "microsoft/Phi-" in llm_id:
        return "<|assistant|>", "<|end|>"
    if "meta-llama/Llama-3" in llm_id:
        return "assistant<|end_header_id|>", "<|eot_id|>"
    if "google/gemma-2" in llm_id:
        return "<start_of_turn>model", "<end_of_turn>"
    if "Qwen2.5" in llm_id:
        return "<|im_start|>assistant", "<|im_end|>"
    return "", ""


def filter_generated_text(text: str, pre_header: str, post_header: str) -> str:
    """Extract only the assistant response between pre_header and post_header."""
    ind_start = text.rfind(pre_header)
    if ind_start == -1:
        return text.strip().lower()
    start = ind_start + len(pre_header)
    only = text[start:]
    ind_suffix = only.find(post_header)
    if ind_suffix != -1:
        only = only[:ind_suffix]
    return only.strip().lower()
