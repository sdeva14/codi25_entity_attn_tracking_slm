"""Shared text normalization for sentences."""
import re

FILTER_PUNCTUATION_TAIL = (".", ":")


def filter_sentence(cur_sent: str) -> str:
    """Normalize sentence: unify spaces, replace/simplify punctuation."""
    if not cur_sent:
        return cur_sent
    cur_sent = cur_sent.replace(",", ", ")
    cur_sent = cur_sent.replace("'", " ")
    cur_sent = cur_sent.replace('"', ", ")
    cur_sent = cur_sent.replace("-", " ")
    cur_sent = cur_sent.replace("/", " ")
    cur_sent = cur_sent.replace("*", " ")
    cur_sent = cur_sent.replace("<", " ")
    cur_sent = cur_sent.replace(">", " ")
    cur_sent = re.sub(r'\.{2,}', '. ', cur_sent)
    cur_sent = re.sub(r'\!{2,}', '! ', cur_sent)
    cur_sent = re.sub(r'\?{2,}', '? ', cur_sent)
    cur_sent = re.sub(r"\s+", " ", cur_sent, flags=re.UNICODE)
    cur_sent = cur_sent.strip()
    if len(cur_sent) > 1 and cur_sent[-1] in FILTER_PUNCTUATION_TAIL:
        cur_sent = cur_sent[:-1]
    return cur_sent
