"""Sentiment label constants and conversions (5-class)."""
import collections
from typing import Union

SENTIMENT_LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
NUM_SENTIMENT_CLASSES = 5

SENTIMENT_PROMPT = (
    "Your role is to classify the sentiment of conversation into 5 classes: "
    "very_positive, positive, neutral, negative, or very_negative. "
    "You must generate only one word of sentiment class."
)

_LABEL_TO_INT = {s: i for i, s in enumerate(SENTIMENT_LABELS)}
_INT_TO_LABEL = {i: s for i, s in enumerate(SENTIMENT_LABELS)}


def label_to_int(label_str: str, default: int = 2) -> int:
    """Map sentiment string to int 0--4. Unknown -> default (neutral)."""
    return _LABEL_TO_INT.get(label_str.strip().lower(), default)


def int_to_label(label_int: int) -> str:
    """Map int 0--4 to sentiment string."""
    return _INT_TO_LABEL.get(label_int, "neutral")


def convert_label_str(sample: dict) -> dict:
    """Convert dataset sample label from int to string (for chat model)."""
    sample["label"] = int_to_label(sample["label"])
    return sample


def convert_label_int(sample: dict, unknown_default: int = 0) -> dict:
    """Convert dataset sample label from string to int. Unknown -> unknown_default."""
    sample["label"] = label_to_int(sample["label"], default=unknown_default)
    return sample


def convert_label_list_int(labels: list) -> list:
    """Convert list of label strings to list of ints. Unknown -> 2 (neutral)."""
    return [label_to_int(curr, default=2) for curr in labels]


def clean_generated_sentiment_class(preds: list) -> list:
    """Normalize model outputs: multi-token or unknown first token -> neutral."""
    cleaned = []
    for curr in preds:
        parts = curr.split()
        if len(parts) > 1 and parts[0] not in SENTIMENT_LABELS:
            cleaned.append("neutral")
        else:
            cleaned.append(curr)
    return cleaned


def get_hist_preds_labels(preds: list, labels: list) -> tuple:
    """Return (Counter(preds), Counter(labels)) for 5-class counts."""
    return collections.Counter(preds), collections.Counter(labels)


def fill_empty_label(hist_curr: dict, num_class: int = NUM_SENTIMENT_CLASSES) -> dict:
    """Ensure keys 0..num_class-1 exist in hist_curr; missing keys set to 0."""
    for i in range(num_class):
        if i not in hist_curr:
            hist_curr[i] = 0
    return hist_curr
