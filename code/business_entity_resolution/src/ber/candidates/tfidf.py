"""Character n-gram TF-IDF with chunked sparse top-k search (task T2.1)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def make_vectorizer(ngram: int, max_df: float) -> TfidfVectorizer:
    """char_wb n-grams: padded per word, so token order matters less.
    ``max_df`` drops the most common n-grams, which dominate search cost and
    carry almost no identity information."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(ngram, ngram), min_df=1,
                           max_df=max_df, sublinear_tf=True, dtype=np.float32)


def fit_index(texts: list[str], ngram: int, max_df: float) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    vec = make_vectorizer(ngram, max_df)
    try:
        X = vec.fit_transform(texts)
    except ValueError:  # empty vocabulary (e.g. every text empty)
        vec = make_vectorizer(ngram, 1.0)
        X = vec.fit_transform([t or " " for t in texts])
    return vec, X.tocsr()


def gpu_available() -> bool:
    """True when CuPy (optional, MIT) and a CUDA device are usable."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # not installed, no driver, no device
        return False


def topk_gpu(Q: sp.csr_matrix, S: sp.csr_matrix, k: int, min_sim: float = 0.0,
             mem_gb: float = 1.5, max_chunk: int = 8192
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU version of :func:`topk` (same contract). S and the query chunk are
    uploaded as CSR; the chunk is densified on the GPU, its similarities against
    every S row are one sparse x dense product (cuSPARSE), and the k best per query
    come from k arg-max passes over that block (cheaper than a sort for small k).
    The chunk size keeps the dense blocks within ``mem_gb``. Only similarities
    > min_sim (and > 0, as the sparse CPU path) are returned; ties at the k-th
    place may be broken differently from the CPU path."""
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n_s = S.shape[0]
    e = np.array([], dtype=np.int64)
    if n_s == 0 or Q.shape[0] == 0:
        return e, e, np.array([], dtype=np.float32)
    kk = min(k, n_s)
    Sg = csp.csr_matrix(S.astype(np.float32))
    per_query = 4 * (n_s + Q.shape[1])       # similarity column + dense query row
    c = int(max(16, min(max_chunk, mem_gb * 1e9 // per_query)))
    thr = max(min_sim, 0.0)
    qi, sj, sim = [], [], []
    for start in range(0, Q.shape[0], c):
        Qc = Q[start:start + c]
        m = Qc.shape[0]
        QdT = csp.csr_matrix(Qc.T.tocsr().astype(np.float32)).toarray()   # (n_feat, m)
        P = Sg @ QdT                                                          # (n_s, m)
        del QdT
        cols = cp.arange(m)
        idx = cp.empty((kk, m), dtype=cp.int64)
        val = cp.empty((kk, m), dtype=cp.float32)
        for r in range(kk):
            j = P.argmax(axis=0)
            idx[r], val[r] = j, P[j, cols]
            P[j, cols] = -1.0
        del P
        keep = val > thr                       # (kk, m), rows already in rank order
        rr, cc_ = cp.nonzero(keep.T)           # query-major order, rank order within
        qi.append(cp.asnumpy(rr).astype(np.int64) + start)
        sj.append(cp.asnumpy(idx.T[rr, cc_]))
        sim.append(cp.asnumpy(val.T[rr, cc_]))
    del Sg
    cp.get_default_memory_pool().free_all_blocks()
    return np.concatenate(qi), np.concatenate(sj), np.concatenate(sim)


def topk(Q: sp.csr_matrix, S: sp.csr_matrix, k: int, chunk: int, n_threads: int,
         min_sim: float = 0.0, device: str = "cpu", gpu_mem_gb: float = 1.5
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k rows of S by cosine for each row of Q (rows already L2-normalised).
    Returns (q_index, s_index, similarity) arrays, ordered by query then by rank.
    ``device="gpu"`` runs :func:`topk_gpu` instead of the multithreaded CPU kernel."""
    if device == "gpu":
        return topk_gpu(Q, S, k, min_sim, gpu_mem_gb)
    ST = S.T.tocsr()
    qi, sj, sim = [], [], []
    for start in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[start:start + chunk], ST, top_n=k, threshold=min_sim,
                           sort=True, n_threads=n_threads).tocsr()
        rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
        qi.append(rows + start)
        sj.append(C.indices.astype(np.int64))
        sim.append(C.data.astype(np.float32))
    if not qi:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    return np.concatenate(qi), np.concatenate(sj), np.concatenate(sim)


def pair_cosine(Q: sp.csr_matrix, S: sp.csr_matrix, qi: np.ndarray, sj: np.ndarray,
                chunk: int = 500_000) -> np.ndarray:
    """Cosine for given (query, index) row pairs, computed in chunks."""
    out = np.empty(len(qi), dtype=np.float32)
    for a in range(0, len(qi), chunk):
        b = min(a + chunk, len(qi))
        out[a:b] = np.asarray(Q[qi[a:b]].multiply(S[sj[a:b]]).sum(axis=1)).ravel()
    return out


def ranks_within_query(qi: np.ndarray) -> np.ndarray:
    """0-based position of each row within its query group (qi must be grouped)."""
    if len(qi) == 0:
        return np.array([], dtype=np.int16)
    starts = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
    counts = np.diff(np.r_[starts, len(qi)])
    return (np.arange(len(qi)) - np.repeat(starts, counts)).astype(np.int16)
