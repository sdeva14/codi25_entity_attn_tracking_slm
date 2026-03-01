# Python Refactoring Review

This document summarizes refactoring opportunities across the repository, ordered by impact and effort.

---

## 1. High impact: duplication and shared logic

### 1.1 `filter_sentence` (text normalization)

**Current:** Same logic appears in three places:

- **attn_flow_test.py** – `filter_sentence()` (lines 38–59) and again **inlined** in the loop (lines 124–140) instead of calling it.
- **ft_attn_flow_test.py** – `filter_sentence()` (lines 419–432).
- **tokenizer_test.py** – same replacements inlined (lines 37–46).

**Refactor:**

- Add a single shared module, e.g. `utils/text_utils.py`, with one `filter_sentence(text: str) -> str`.
- In **attn_flow_test.py**, delete the inlined block and call `filter_sentence(cur_sent)` in the loop.
- In **ft_attn_flow_test.py** and **tokenizer_test.py**, import and use the shared function.
- Optionally add minimal tests for edge cases (empty, punctuation-only).

---

### 1.2 Sentiment labels and label ↔ int mapping

**Current:** The 5-class sentiment set and conversions are repeated many times:

- Literal list: `["very_negative", "negative", "neutral", "positive", "very_positive"]` in `clean_generated_sentiment_class`, and in prompts.
- **convert_label_str** / **convert_label_int** (sample-level) and **convert_label_list_int** (list-level) repeat the same if/elif chains.
- **convert_pred_numeric** (in ft_attn_flow_test.py) uses a different pattern (`curr.name`) and the same 0–4 mapping.

**Refactor:**

- Define in one place, e.g. `utils/sentiment.py`:
  - `SENTIMENT_LABELS: list[str]` (single source of truth).
  - `label_to_int(s: str) -> int` and `int_to_label(i: int) -> str`.
- Replace all if/elif chains and literal lists with these helpers.
- Use the same mapping for sample conversion, list conversion, and pred conversion so behavior is consistent and easy to change (e.g. 3-class later).

---

### 1.3 `get_avg_from_lists` name/signature clash

**Current:**

- **attn_flow_test.py** / **ft_attn_flow_test.py**: `get_avg_from_lists(list_val)` → returns `[mean, std]` (or list of two floats) over a nested or flat list of numbers.
- **stat_analysis.py**: `get_avg_from_lists(output_dir, sub_dir, output_name)` → reads a JSON log file and returns `(mean, std)`.

Same name, different signatures and return shapes; easy to misuse.

**Refactor:**

- **stat_analysis**: Rename to e.g. `read_log_and_compute_stats(output_dir, sub_dir, output_name) -> tuple[float, float]`, and implement by reading JSON, flattening, then calling a shared numeric helper.
- Add a single implementation for “flatten list(s) and compute mean/std”, e.g. in `utils/stats.py`: `mean_std(values: list) -> tuple[float, float]`, and use it from both scripts and from stat_analysis.

---

### 1.4 LLM response headers (inference_sentiment)

**Current:** In **ft_attn_flow_test.py** `inference_sentiment`, the `pre_header` / `post_header` for stripping model output are chosen with if/elif on `llm_id` (Phi, Llama, Gemma, Qwen).

**Refactor:**

- Move to a small map or function, e.g. in `utils/llm_utils.py`: `get_response_delimiters(llm_id: str) -> tuple[str, str]`.
- Call it once per inference (or once per run if llm_id is fixed). Makes adding new models and testing easier.

---

### 1.5 `load_dataset_toefl` and stanza sentence splitting

**Current:** Very similar logic in:

- **attn_flow_test.py** – `load_dataset_toefl` (DATA_PATH, CSV fold, stanza, concat).
- **ft_attn_flow_test.py** – `load_dataset_toefl` (cfg.dataset.path_data, CSV fold, stanza, concat) plus SST-5 path in the same file.

**Refactor:**

- Implement one function, e.g. in `corpus/dataset_toefl_hf.py` or a new `corpus/load_toefl.py`, that takes `(path_data, num_samples, target_prompt_or_score)` and returns `(sent_corpus, num_sents)`.
- Use `os.environ.get("DATA_PATH") or path_data` inside that module so both entrypoints (attn_flow_test vs ft) can share it.
- Keep SST-5 loading separate (different dataset) but in the same corpus package if desired.

---

## 2. Medium impact: structure and file layout

### 2.1 Split `ft_attn_flow_test.py`

**Current:** One large file (~740 lines) that mixes:

- PEFT/LoRA training (`peft_encoder`, SFTTrainer, training args).
- Sentiment inference and eval (`inference_sentiment`, `clean_generated_sentiment_class`, `eval_accuracy`, `eval_kappa`, etc.).
- Entity attention flow and aggregation (`wrapper_ent_attn_flow`, `get_avg_from_lists`, logging).
- Dataset loading and sampling (`load_dataset_toefl`, `sample_dataset`).
- Plotting (`plot_preds_labels_dists`, matplotlib).
- Hydra `main()` and CLI.

**Refactor:**

- Extract to separate modules, e.g.:
  - **sentiment_utils.py** (or under `utils/`): label constants, `convert_label_*`, `clean_generated_sentiment_class`, `filter_generated_text`, `get_response_delimiters`, `inference_sentiment`, `eval_accuracy`, `eval_kappa`, `get_hist_preds_labels`, `fill_empty_label`.
  - **plot_utils.py**: `plot_preds_labels_dists` (and any other plotting used only for sentiment/attn).
  - **sampling.py** or under corpus: `sample_dataset`, and optionally `load_dataset_toefl` once unified.
- Keep in **ft_attn_flow_test.py**: `main()`, `peft_encoder`, `wrapper_ent_attn_flow`, `naive_inference`, `ent_attn_layer_analysis`, and imports from the new modules.
- This improves readability, reuse, and testing of sentiment and plotting in isolation.

---

### 2.2 Unify `wrapper_ent_attn_flow` and aggregation

**Current:**

- **attn_flow_test.py**: corpus is “list of documents” (each doc = list of sentences). Loop over docs, then over sentences; aggregate per-doc lists then flatten for stats; `num_sents_all` passed in.
- **ft_attn_flow_test.py**: corpus is a pandas DataFrame (one row per sentence). Single loop over rows; aggregate lists directly; `num_sents_all = len(sent_corpus)`.

A lot of the inner logic (run_attn_flow, append to list_val_* type1–4, len_subwords, num_entities, logging) is duplicated.

**Refactor:**

- Extract a single function that, given one sentence and config/tokenizer/encoder/parser, returns the same structure as `ent_attn_flow.run_attn_flow` (or a small wrapper around it).
- Extract an “aggregator” that:
  - Takes a stream of per-sentence results (or list of them).
  - Updates running lists (type1–4, ent_topk, vp_topk, lengths, counts).
  - Optionally logs progress.
- Each script then:
  - Builds the appropriate “stream” (by doc/sent or by row).
  - Calls the aggregator and then the same “print summary / return outputs” block.
- This removes duplicated aggregation and reporting and keeps behavior consistent.

---

## 3. Bug fixes and robustness

### 3.1 `attn_flow_test.py`: undefined `list_np_sbw_loc` / `list_vp_sbw_loc`

**Current:** In the sentence loop, when `is_parsed_all and is_vp_parsed_all` is false, the code goes to `else: num_sents_corrupted += 1`. Right after the if/else, it does:

```python
if len(list_np_sbw_loc) < 1:    num_sents_non_ent += 1
if len(list_vp_sbw_loc) < 1:    num_sents_no_vp += 1
```

In the else branch, `list_np_sbw_loc` and `list_vp_sbw_loc` were never set → possible `NameError`.

**Refactor:** Same as in ft_attn_flow_test.py: after the if/else, set:

```python
list_np_sbw_loc = output_attn_flow.get("list_np_sbw_loc", [])
list_vp_sbw_loc = output_attn_flow.get("list_vp_sbw_loc", [])
```

then run the two `len(...)` checks. Optionally add a one-line comment that these are for counting only.

---

### 3.2 `attn_flow_test.py`: use `filter_sentence` in the loop

**Current:** Lines 124–140 duplicate the body of `filter_sentence` inside the loop instead of calling it.

**Refactor:** Replace that block with `cur_sent = filter_sentence(cur_sent)`.

---

### 3.3 `attn_flow_test.py`: top-level `import re`

**Current:** `re` is used in `filter_sentence` (e.g. `re.sub`, `re.UNICODE`) but `import re` is only done inside `wrapper_ent_attn_flow`. It works but is inconsistent.

**Refactor:** Add `import re` at the top of the file with the other standard-library imports.

---

## 4. Code quality and maintainability

### 4.1 Entity parser

- **np_parser_backt.py**
  - **Typo:** `filter_speical_tokens` → `filter_special_tokens` (and update all call sites).
  - **filter_sbw_str:** Long if-chains for leading/trailing characters. Consider sets, e.g. `PREFIX_CHARS = {",", "?", "!", "(", "\"", "'", "["}` and `s = s.lstrip(...)` / `s = s.rstrip(...)` or a small loop, to make adding/removing characters easier.
  - **extract_np_with_len** and **extract_vp_with_len** are almost identical; only the tag filter differs. Consider a single `_extract_phrases_with_len(sent, max_len, tag_predicate)` where `tag_predicate(tag)` is e.g. `"NP" in tag or "NN" in tag` vs `"VB" in tag`.

---

### 4.2 ent_attn_func/attn_flow.py

- **attn_flow_entity** builds type1–type4 flow sums and is not used by **run_attn_flow** (which uses **attn_ranking_entity**). Either remove it as dead code or document that it is kept for alternative metrics/experiments; if kept, consider moving to a separate “legacy” or “experimental” helper so the main flow stays clear.
- **flat_index_to_list** exists but **attn_flow_entity** reimplements the same logic in a loop. Use `flat_index_to_list(ind_ent)` there for consistency and less duplication.

---

### 4.3 Unused / dead code in ft_attn_flow_test.py `main()`

**Current:** Variables are set but not used when loading the encoder:

- `use_4bit`, `use_nested_quant`, `bnb_4bit_compute_dtype`, `bnb_4bit_quant_type`, `compute_dtype`, `bnb_config` (BitsAndBytesConfig) are computed, but the model is loaded with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=encoder_dtype)` and no `quantization_config=bnb_config`.

**Refactor:** Either:

- Use 4-bit loading: pass `bnb_config` (and device_map etc.) into `from_pretrained`, and remove the unused `encoder_dtype` path for that case, or
- If you do not intend to use 4-bit in this script, remove the bnb_* and BitsAndBytesConfig block and the unused `compute_dtype`/fp16/bf16 if they are only for that path, to avoid confusion.

---

### 4.4 Corpus and collators

- **corpus/dataset_toefl_hf.py**
  - **filter_special_tokens** only reassigns the same keys; it has no effect. Either implement real filtering or remove it (and any callers) to avoid confusion.
  - **__init__**: The trailing `return` is unnecessary in a constructor; can be removed for style.
- **collators/collator_toefl.py**
  - The final block that maps `"label"` / `"label_ids"` to `"labels"` can be simplified to a single small helper or a couple of lines that normalize the key once.

---

## 5. Minor improvements

- **Imports:** In ft_attn_flow_test.py, trim duplicate or unused imports (e.g. `logging` twice, `numpy` twice, `Path` if unused, `json`/`time` if only used in one place). Group and order (stdlib → third-party → local) for consistency.
- **Constants:** Centralize magic numbers and strings (e.g. number of classes 5, default top_k 5, sample_ratio 0.1, log dir names, file names like `"type1.log"`) in a small `config` or `constants` module or at the top of the relevant module.
- **Type hints:** Add gradual typing to public functions (e.g. `filter_sentence`, label converters, `run_attn_flow`, `get_avg_from_lists`/stats helpers). Start with the most reused and CLI-facing functions.
- **Docstrings:** Add one-line (or short) docstrings to functions that are shared or non-obvious (e.g. what “type1”–“type4” mean in attention flow).

---

## Suggested order of work

1. **Quick wins:** Fix attn_flow_test.py bug (3.1), use `filter_sentence` in loop (3.2), add `import re` (3.3).  
2. **Shared text/sentiment:** Introduce `utils/text_utils.py` and `utils/sentiment.py`, then refactor `filter_sentence` and label logic (1.1, 1.2).  
3. **Stats and naming:** Unify and rename get_avg / read-log logic (1.3).  
4. **LLM and dataset:** Centralize response delimiters (1.4) and TOEFL loading (1.5).  
5. **Structure:** Split ft_attn_flow_test.py (2.1) and unify wrapper + aggregation (2.2).  
6. **Cleanup:** Parser typo and helpers (4.1), attn_flow dead code / flat_index reuse (4.2), bnb/corpus/collator (4.3, 4.4), then constants and typing (5).

If you tell me which part you want to tackle first (e.g. “only bugs and filter_sentence” or “full sentiment utils”), I can outline concrete patches or diffs for that part.
