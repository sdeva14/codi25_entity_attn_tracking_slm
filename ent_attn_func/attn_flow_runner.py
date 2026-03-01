"""
Shared entity attention-flow aggregation and logging.
Used by attn_flow_test.py (doc-level corpus) and ft_attn_flow_test.py (sentence-level DataFrame).
"""
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, List

from utils.stats import get_avg_from_lists

MIN_WORDS_SENTENCE = 5
LOG_DIR = "log_entity_attn"
TYPE_LOG_NAMES = ("type1.log", "type2.log", "type3.log", "type4.log")


@dataclass
class AttnFlowAggregator:
    """Holds running counts and lists for entity-attn flow over a corpus."""
    # Per-sentence / per-doc value lists (nested when corpus_style="doc")
    list_val_ent_topk: List[List[float]] = field(default_factory=list)
    list_val_vp_topk: List[List[float]] = field(default_factory=list)
    list_val_interact_type1: List[List[float]] = field(default_factory=list)
    list_val_interact_type2: List[List[float]] = field(default_factory=list)
    list_val_interact_type3: List[List[float]] = field(default_factory=list)
    list_val_interact_type4: List[List[float]] = field(default_factory=list)
    # Counts
    num_sents_non_ent: int = 0
    num_sents_corrupted: int = 0
    num_sents_short: int = 0
    num_sents_no_vp: int = 0
    num_sents_corrupted_vp: int = 0
    num_sents_considered: int = 0
    # Lengths (for logging)
    list_len_subwords: List[int] = field(default_factory=list)
    list_len_subwords_ent: List[int] = field(default_factory=list)
    list_len_subwords_vp: List[int] = field(default_factory=list)
    list_num_entities: List[int] = field(default_factory=list)
    # Cache for entity parsing (doc-level)
    list_doc_np_sbw_loc: List[List[Any]] = field(default_factory=list)
    list_doc_vp_sbw_loc: List[List[Any]] = field(default_factory=list)
    # Doc-mode: current doc accumulators
    _doc_val_ent_topk: List[float] = field(default_factory=list, repr=False)
    _doc_val_vp_topk: List[float] = field(default_factory=list, repr=False)
    _doc_val_type1: List[float] = field(default_factory=list, repr=False)
    _doc_val_type2: List[float] = field(default_factory=list, repr=False)
    _doc_val_type3: List[float] = field(default_factory=list, repr=False)
    _doc_val_type4: List[float] = field(default_factory=list, repr=False)
    _doc_np_sbw_loc: List[Any] = field(default_factory=list, repr=False)
    _doc_vp_sbw_loc: List[Any] = field(default_factory=list, repr=False)
    _doc_mode: bool = False

    @classmethod
    def for_doc_corpus(cls) -> "AttnFlowAggregator":
        """Use when corpus is list of documents (each doc = list of sentences)."""
        a = cls()
        a._doc_mode = True
        return a

    @classmethod
    def for_flat_corpus(cls) -> "AttnFlowAggregator":
        """Use when corpus is flat (e.g. DataFrame with one row per sentence)."""
        a = cls()
        a._doc_mode = False
        return a

    def update(self, output_attn_flow: dict, list_np_sbw_loc: list, list_vp_sbw_loc: list) -> None:
        """Update aggregator from one sentence's run_attn_flow output."""
        is_parsed = output_attn_flow.get("is_parsed_all") and output_attn_flow.get("is_vp_parsed_all")
        if is_parsed:
            ent_top_k = output_attn_flow.get("ent_top_k")
            vp_top_k = output_attn_flow.get("vp_top_k")
            t1, t2, t3, t4 = output_attn_flow.get("val_type1"), output_attn_flow.get("val_type2"), output_attn_flow.get("val_type3"), output_attn_flow.get("val_type4")
            seq_len = output_attn_flow.get("seq_len_subwords")
            seq_len_ent = output_attn_flow.get("len_subwords_ent")
            seq_len_vp = output_attn_flow.get("len_subwords_vp")
            np_loc = output_attn_flow.get("list_np_sbw_loc", [])
            vp_loc = output_attn_flow.get("list_vp_sbw_loc", [])

            if self._doc_mode:
                if ent_top_k is not None and not math.isnan(ent_top_k):
                    self._doc_val_ent_topk.append(ent_top_k)
                if vp_top_k is not None and not math.isnan(vp_top_k):
                    self._doc_val_vp_topk.append(vp_top_k)
                if t1 is not None and not math.isnan(t1):
                    self._doc_val_type1.append(t1)
                if t2 is not None and not math.isnan(t2):
                    self._doc_val_type2.append(t2)
                if t3 is not None and not math.isnan(t3):
                    self._doc_val_type3.append(t3)
                if t4 is not None and not math.isnan(t4):
                    self._doc_val_type4.append(t4)
                self._doc_np_sbw_loc.append(np_loc)
                self._doc_vp_sbw_loc.append(vp_loc)
            else:
                if ent_top_k is not None and not math.isnan(ent_top_k):
                    self.list_val_ent_topk.append([ent_top_k])
                if vp_top_k is not None and not math.isnan(vp_top_k):
                    self.list_val_vp_topk.append([vp_top_k])
                if t1 is not None and not math.isnan(t1):
                    self.list_val_interact_type1.append([t1])
                if t2 is not None and not math.isnan(t2):
                    self.list_val_interact_type2.append([t2])
                if t3 is not None and not math.isnan(t3):
                    self.list_val_interact_type3.append([t3])
                if t4 is not None and not math.isnan(t4):
                    self.list_val_interact_type4.append([t4])
                self.list_doc_np_sbw_loc.append(np_loc)
                self.list_doc_vp_sbw_loc.append(vp_loc)

            self.list_num_entities.append(len(np_loc))
            self.list_len_subwords.append(seq_len)
            self.list_len_subwords_ent.append(seq_len_ent)
            self.list_len_subwords_vp.append(seq_len_vp)
            self.num_sents_considered += 1
        else:
            self.num_sents_corrupted += 1

        if len(list_np_sbw_loc) < 1:
            self.num_sents_non_ent += 1
        if len(list_vp_sbw_loc) < 1:
            self.num_sents_no_vp += 1

    def end_document(self) -> None:
        """Call at end of each document when in doc mode. Flushes current doc into list_val_*."""
        if not self._doc_mode:
            return
        if self._doc_val_ent_topk:
            self.list_val_ent_topk.append(self._doc_val_ent_topk)
        if self._doc_val_vp_topk:
            self.list_val_vp_topk.append(self._doc_val_vp_topk)
        if self._doc_val_type1:
            self.list_val_interact_type1.append(self._doc_val_type1)
        if self._doc_val_type2:
            self.list_val_interact_type2.append(self._doc_val_type2)
        if self._doc_val_type3:
            self.list_val_interact_type3.append(self._doc_val_type3)
        if self._doc_val_type4:
            self.list_val_interact_type4.append(self._doc_val_type4)
        self.list_doc_np_sbw_loc.append(self._doc_np_sbw_loc)
        self.list_doc_vp_sbw_loc.append(self._doc_vp_sbw_loc)
        self._doc_val_ent_topk = []
        self._doc_val_vp_topk = []
        self._doc_val_type1 = []
        self._doc_val_type2 = []
        self._doc_val_type3 = []
        self._doc_val_type4 = []
        self._doc_np_sbw_loc = []
        self._doc_vp_sbw_loc = []

    def log_summary(self, num_sents_all: int, logger: Any) -> None:
        """Log aggregate stats (subword lengths, counts, Ent@K, VP@K, Type1--4)."""
        if self.list_len_subwords:
            logger.info(f"Avg # of subwords in a sentence: {statistics.mean(self.list_len_subwords)}")
            logger.info(f"Avg # of all subwords for entities in a sentence: {statistics.mean(self.list_len_subwords_ent)}")
            logger.info(f"Avg # of all subwords for verbs in a sentence: {statistics.mean(self.list_len_subwords_vp)}")
        if self.list_num_entities:
            logger.info(f"Avg # of entities in a sentence: {statistics.mean(self.list_num_entities)}")
        logger.info("")
        logger.info(f"# of sentences without entities: {self.num_sents_non_ent} ({self.num_sents_non_ent / max(1, num_sents_all):.2%})")
        logger.info(f"# of sentences not properly parsed: {self.num_sents_corrupted} ({100 * self.num_sents_corrupted / max(1, num_sents_all):.2f}%)")
        logger.info(f"# of sentences not considered due to short: {self.num_sents_short} ({100 * self.num_sents_short / max(1, num_sents_all):.2f}%)")
        logger.info(f"# of sentences considered in total: {self.num_sents_considered} ({100 * self.num_sents_considered / max(1, num_sents_all):.2f}%)")
        logger.info("")
        logger.info(f"# of sentences without VP: {self.num_sents_no_vp} ({self.num_sents_no_vp / max(1, num_sents_all):.2%})")
        logger.info(f"# of sentences not properly VP parsed: {self.num_sents_corrupted_vp} ({100 * self.num_sents_corrupted_vp / max(1, num_sents_all):.2f})")
        logger.info("")

        for name, list_val in [
            ("Ent@K", self.list_val_ent_topk),
            ("VP@K", self.list_val_vp_topk),
        ]:
            if list_val:
                out = get_avg_from_lists(list_val)
                logger.info(f"{name} Avg(Std): {out[0]} ({out[1]})")
        logger.info("")

        avg_stds = []
        for i, list_val in enumerate(
            [self.list_val_interact_type1, self.list_val_interact_type2, self.list_val_interact_type3, self.list_val_interact_type4],
            start=1,
        ):
            if list_val:
                out = get_avg_from_lists(list_val)
                avg_stds.append((float(out[0]), float(out[1])))
                logger.info(f"Type{i} Avg(Std): {out[0]} ({out[1]})")
            else:
                avg_stds.append((0.0, 0.0))
        logger.info("")
        self._last_avg_stds = avg_stds

    def build_results(self) -> List:
        """Return the same list structure as before: [list1, list2, list3, list4, (avg1,std1), ...]."""
        avg_stds = getattr(self, "_last_avg_stds", None)
        if avg_stds is None:
            avg_stds = []
            for list_val in [
                self.list_val_interact_type1,
                self.list_val_interact_type2,
                self.list_val_interact_type3,
                self.list_val_interact_type4,
            ]:
                if list_val:
                    out = get_avg_from_lists(list_val)
                    avg_stds.append((float(out[0]), float(out[1])))
                else:
                    avg_stds.append((0.0, 0.0))
        result = [
            self.list_val_interact_type1,
            self.list_val_interact_type2,
            self.list_val_interact_type3,
            self.list_val_interact_type4,
            *avg_stds,
        ]
        return result


def write_logs(output_dir: str, sub_dir: str, results: List, logger: Any) -> None:
    """Write type1--type4 list values to output_dir/sub_dir/typeN.log. results from build_results()."""
    import os
    import json
    path_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(path_dir, exist_ok=True)
    for i, name in enumerate(TYPE_LOG_NAMES):
        if i < 4 and i < len(results):
            with open(os.path.join(path_dir, name), "w") as f:
                json.dump(results[i], f)
    logger.info("")
