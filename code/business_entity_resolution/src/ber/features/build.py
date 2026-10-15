"""Pair + context features for candidate pairs (tasks T3.1–T3.3).

Input: candidate pairs joined with both records' normalised fields, columns
prefixed ``q_`` (the S2/S3 query) and ``s_`` (the S1 candidate). String
similarities use ``rapidfuzz.process.cpdist`` (vectorised, multithreaded).

Groups (see eda/FINDINGS.md):
  * name: exact/compact equality, edit distances split into substitutions vs
    insertions/deletions vs transpositions (distractors change letters,
    true-pair noise mostly drops/doubles/swaps them), fuzzy ratios, alias parts,
    initials, legal-form agreement/conflict, gibberish (vocabulary OOV share),
    name ambiguity (how many S1 share the core name)
  * address: equality, fuzzy ratios, region agreement, house-number relation
    (equal/padded/truncated vs shifted), number-set overlap, PO box/PMB
  * retrieval: TF-IDF cosines and ranks from stage 1
  * context: each score relative to the query's other candidates (margin over
    the best other candidate, rank, gap to best), candidate counts per query
    and per S1
No country feature: every feature means the same thing in any country.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Indel, JaroWinkler, Levenshtein, OSA

from .numbers import number_features, number_features_v2

CONTEXT_COLS = ["cos_combo", "cos_name", "cos_addr", "n_tsr", "n_jw", "a_tsr"]
KEY_CONTEXT_COLS = ["k_ratio", "k_idf_rec"]  # with key features (v2)


def _cp(a: pd.Series, b: pd.Series, scorer) -> np.ndarray:
    return process.cpdist(a.tolist(), b.tolist(), scorer=scorer, workers=-1)


def initials(core: pd.Series) -> pd.Series:
    return core.map(lambda s: "".join(t[0] for t in s.split()) if s else "")


def name_features(d: pd.DataFrame) -> pd.DataFrame:
    qc, sc = d.q_name_core, d.s_name_core
    qx, sx = d.q_name_compact, d.s_name_compact
    f = pd.DataFrame(index=d.index)
    f["n_core_eq"] = (qc == sc).astype(np.int8)
    f["n_compact_eq"] = (qx == sx).astype(np.int8)
    lev, osa, indel = _cp(qx, sx, Levenshtein.distance), _cp(qx, sx, OSA.distance), \
        _cp(qx, sx, Indel.distance)
    f["n_lev"] = lev
    f["n_sub"] = indel - lev          # substitutions on the optimal path
    f["n_transp"] = lev - osa         # adjacent swaps
    f["n_len_q"], f["n_len_s"] = qx.str.len(), sx.str.len()
    f["n_len_diff"] = f.n_len_q - f.n_len_s
    f["n_ratio"] = _cp(qc, sc, fuzz.ratio)
    f["n_tsr"] = _cp(qc, sc, fuzz.token_set_ratio)
    f["n_tsort"] = _cp(qc, sc, fuzz.token_sort_ratio)
    f["n_partial"] = _cp(qc, sc, fuzz.partial_ratio)
    f["n_jw"] = _cp(qx, sx, JaroWinkler.similarity)
    alias = d.q_name_has_alias.to_numpy()
    alt = np.maximum(_cp(d.q_name_alt_core, sc, fuzz.ratio), _cp(d.q_name_pre_core, sc, fuzz.ratio))
    f["n_alias_best"] = np.where(alias, alt, -1)
    f["n_first_eq"] = (qc.str.split().str[0] == sc.str.split().str[0]).astype(np.int8)
    f["n_last_eq"] = (qc.str.split().str[-1] == sc.str.split().str[-1]).astype(np.int8)
    f["n_q_is_initials"] = ((qx == d.s_initials) & (qx.str.len() >= 2)).astype(np.int8)
    f["n_s_is_initials"] = ((sx == d.q_initials) & (sx.str.len() >= 2)).astype(np.int8)
    ql, sl = d.q_name_legal, d.s_name_legal
    f["l_eq"] = ((ql == sl) & (ql != "")).astype(np.int8)
    f["l_conflict"] = [int(bool(a) and bool(b) and not set(a.split()) & set(b.split()))
                       for a, b in zip(ql, sl)]
    f["l_q_empty"], f["l_s_empty"] = (ql == "").astype(np.int8), (sl == "").astype(np.int8)
    f["q_is_domain"] = d.q_name_is_domain.astype(np.int8)
    f["q_has_alias"] = d.q_name_has_alias.astype(np.int8)
    f["q_indic"] = d.q_name_indic.astype(np.int8)
    f["q_indic_oov"] = d.q_name_n_indic_oov
    f["q_vocab_oov"] = d.q_vocab_oov
    f["s_name_freq"] = d.s_name_freq
    if "q_brand" in d:  # v2: query name is a made-up brand (only the address can match it)
        f["q_brand"] = d.q_brand.astype(np.int8)
        f["s_brand"] = d.s_brand.astype(np.int8)
    return f


class KeyIndex:
    """Sparse token matrices over the filler-free name keys of one split's records
    (row = record position ``kpos``): ``B`` binary, ``W`` IDF-weighted (IDF from
    S1 keys). Makes the IDF overlap features vectorised."""

    def __init__(self, keys: pd.Series, idf: dict[str, float]):
        vocab: dict[str, int] = {}
        indptr, indices = [0], []
        for k in keys:
            for t in k.split():
                indices.append(vocab.setdefault(t, len(vocab)))
            indptr.append(len(indices))
        top = max(idf.values(), default=1.0)
        w = np.array([idf.get(t, top) for t in vocab], dtype=np.float32)
        shape = (len(keys), max(len(vocab), 1))
        ind = np.asarray(indices, dtype=np.int32)
        self.B = sp.csr_matrix((np.ones(len(ind), np.float32), ind, indptr), shape=shape)
        self.W = sp.csr_matrix((w[ind] if len(ind) else ind.astype(np.float32), ind, indptr),
                               shape=shape)
        self.wsum = np.asarray(self.W.sum(axis=1)).ravel()
        self.n = np.diff(self.B.indptr).astype(np.float32)


def _rowmax(m: sp.csr_matrix) -> np.ndarray:
    return np.asarray(m.max(axis=1).todense()).ravel()


def key_features(d: pd.DataFrame, kidx: KeyIndex, chunk: int = 500_000) -> pd.DataFrame:
    """Filler-free name keys (sorted unique core tokens without injected filler
    words, see features.stage.learn_fillers) and IDF-weighted token overlap.

    Names in true pairs often differ by one injected filler ("Safe Colonial
    Airlines" -> "Safe Colonial Center") or by word order ("V Cascade
    Fellowship"); the key removes both. The IDF overlap tells a shared rare token
    ("Tavosoljax") from a shared generic one ("Clinic"), and the largest IDF
    among unmatched tokens flags a distinctive word present on one side only
    ("Meridian Dental" vs "Meridian Plumbing")."""
    qk, sk = d.q_name_key, d.s_name_key
    f = pd.DataFrame(index=d.index)
    f["k_eq"] = (qk == sk).astype(np.int8)
    qx, sx = qk.str.replace(" ", "", regex=False), sk.str.replace(" ", "", regex=False)
    f["k_lev"] = _cp(qx, sx, Levenshtein.distance)
    f["k_ratio"] = _cp(qk, sk, fuzz.ratio)
    f["k_tsr"] = _cp(qk, sk, fuzz.token_set_ratio)
    out = idf_overlap(kidx, d.q_kpos.to_numpy(np.int64), d.s_kpos.to_numpy(np.int64), chunk)
    for j, c in enumerate(IDF_COLS):
        f[f"k_{c}"] = out[:, j]
    return f


IDF_COLS = ["idf_rec", "idf_prec", "idf_unmatched_s", "idf_unmatched_q", "n_unmatched_q",
            "n_unmatched_s"]


def idf_overlap(kidx: KeyIndex, qp: np.ndarray, spos: np.ndarray,
                chunk: int = 500_000) -> np.ndarray:
    """Per pair (query row qp, S1 row spos of ``kidx``): IDF-weighted share of the
    S1 tokens found in the query (recall) and of the query tokens found in the S1
    (precision), the largest IDF of an unmatched S1 / query token, and the
    numbers of unmatched query / S1 tokens (columns of IDF_COLS)."""
    out = np.zeros((len(qp), 6), dtype=np.float32)
    B, W = kidx.B, kidx.W
    for a in range(0, len(qp), chunk):
        b = min(a + chunk, len(qp))
        q, s = qp[a:b], spos[a:b]
        Bq, Bs, Wq, Ws = B[q], B[s], W[q], W[s]
        Is, Iq = Ws.multiply(Bq).tocsr(), Wq.multiply(Bs).tocsr()
        wi = np.asarray(Is.sum(axis=1)).ravel()
        n_i = np.asarray(Bq.multiply(Bs).sum(axis=1)).ravel()
        wa, wb = kidx.wsum[q], kidx.wsum[s]
        out[a:b, 0] = np.divide(wi, wb, out=np.zeros_like(wi), where=wb > 0)
        out[a:b, 1] = np.divide(wi, wa, out=np.zeros_like(wi), where=wa > 0)
        out[a:b, 2] = _rowmax((Ws - Is).tocsr())
        out[a:b, 3] = _rowmax((Wq - Iq).tocsr())
        out[a:b, 4] = kidx.n[q] - n_i
        out[a:b, 5] = kidx.n[s] - n_i
    return out


def address_features(d: pd.DataFrame) -> pd.DataFrame:
    qa, sa = d.q_addr_norm, d.s_addr_norm
    f = pd.DataFrame(index=d.index)
    f["a_q_empty"], f["a_s_empty"] = (qa == "").astype(np.int8), (sa == "").astype(np.int8)
    f["a_bag_eq"] = ((d.q_addr_bag == d.s_addr_bag) & (qa != "")).astype(np.int8)
    f["a_tsr"] = _cp(qa, sa, fuzz.token_set_ratio)
    f["a_ratio"] = _cp(d.q_addr_bag, d.s_addr_bag, fuzz.ratio)
    f["a_partial"] = _cp(qa, sa, fuzz.partial_ratio)
    qr, sr = d.q_region, d.s_region
    f["r_eq"] = ((qr == sr) & (qr != "")).astype(np.int8)
    f["r_conflict"] = ((qr != sr) & (qr != "") & (sr != "")).astype(np.int8)
    f["r_q_missing"] = (qr == "").astype(np.int8)
    nums = [number_features(a, b) for a, b in zip(d.q_addr_numbers, d.s_addr_numbers)]
    nf = pd.DataFrame(nums, index=d.index,
                      columns=["hn_rel", "hn_logdiff", "num_common", "num_q_only",
                               "num_s_only", "num_jacc"])
    nums2 = [number_features_v2(a, b) for a, b in zip(d.q_addr_numbers, d.s_addr_numbers)]
    nf2 = pd.DataFrame(nums2, index=d.index, dtype=np.int8,
                       columns=["hn_s_in_q", "hn_q_in_s", "hn_best_rel", "hn_lev", "hn_diff_pos"])
    f = pd.concat([f, nf, nf2], axis=1)
    f["hn_rel"] = f.hn_rel.astype(np.int8)
    f["box_q"], f["box_s"] = d.q_addr_has_box.astype(np.int8), d.s_addr_has_box.astype(np.int8)
    return f


def address_key(bag: pd.Series) -> pd.Series:
    """Address tokens without numbers (numbers have their own features)."""
    return bag.map(lambda b: " ".join(t for t in b.split() if not t.isdigit()))


def address_idf_features(d: pd.DataFrame, aidx: KeyIndex) -> pd.DataFrame:
    """v2: IDF-weighted overlap of the address words (street and locality names
    are rare, city and street types common), for pairs whose names can't help
    (made-up brands, initials) and for lookalikes at a different street."""
    out = idf_overlap(aidx, d.q_kpos.to_numpy(np.int64), d.s_kpos.to_numpy(np.int64))
    return pd.DataFrame({f"a_{c}": out[:, j] for j, c in enumerate(IDF_COLS)}, index=d.index)


def _best_other(q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows grouped by query id ``q``: best value in the group, best value
    among the *other* rows of the group, and rank (0 = best) of each row."""
    order = np.lexsort((-v, q))
    qs, vs = q[order], v[order]
    starts = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(qs)])
    first = np.repeat(vs[starts], counts)
    second_idx = np.minimum(starts + 1, len(vs) - 1)
    second = np.where(counts > 1, vs[second_idx], np.nan)
    second = np.repeat(second, counts)
    rank = np.arange(len(qs)) - np.repeat(starts, counts)
    other = np.where(rank == 0, second, first)
    out_best, out_other, out_rank = (np.empty_like(first), np.empty_like(other),
                                     np.empty(len(v), dtype=np.int16))
    out_best[order], out_other[order], out_rank[order] = first, other, rank
    return out_best, out_other, out_rank


def context_features(d: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Scores relative to the same query's other candidates. ``d`` must be sorted
    or at least contain every candidate of each query it holds."""
    q = pd.factorize(d.q_id)[0]
    out = pd.DataFrame(index=d.index)
    for col in CONTEXT_COLS + [c for c in KEY_CONTEXT_COLS if c in f]:
        v = (d[col] if col in d else f[col]).to_numpy(dtype=np.float64)
        best, other, rank = _best_other(q, v)
        out[f"{col}_margin"] = v - np.nan_to_num(other, nan=0.0)   # >0: beats all others
        out[f"{col}_gap"] = best - v
        out[f"{col}_rank"] = rank
    out["q_n_cands"] = np.bincount(q)[q]
    return out


def build_features(d: pd.DataFrame, kidx: KeyIndex | None = None,
                   aidx: KeyIndex | None = None) -> pd.DataFrame:
    """All features for a frame of candidate pairs with q_/s_ record fields,
    stage-1 retrieval columns, and per-S1 context (s_n_cands, s_n_top1). With
    ``kidx`` / ``aidx`` (and q_/s_name_key, q_/s_kpos columns) also the v2 name-key
    and address-word IDF features."""
    retr = d[["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo",
              "s_n_cands", "s_n_top1"]].copy()
    retr["q_source"] = d.q_source.astype(np.int8)
    nf = name_features(d)
    af = address_features(d)
    parts = [retr, nf, af]
    if kidx is not None and "q_name_key" in d:
        parts.append(key_features(d, kidx))
    if aidx is not None and "q_kpos" in d:
        parts.append(address_idf_features(d, aidx))
    f = pd.concat(parts, axis=1)
    return pd.concat([f, context_features(d, f)], axis=1)
